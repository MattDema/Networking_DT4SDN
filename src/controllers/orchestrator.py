import os
import sys
import time
import threading
import requests
import numpy as np

# -- GETTING PATHS TO ACCESS CLASSES INSTANCES --
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)
project_root = os.path.dirname(src_dir)

from database.db_manager import get_db
# CHANGED: Import StatePredictor instead of TrafficPredictor
from ml_models.state_predictor import StatePredictor
from web_interface.app import start_web_server, get_active_switches, get_hosts
from utils.traffic_manager import TrafficManager
import numpy as np
from scipy.stats import linregress


# =============================================================================
# CONFIGURATION
# =============================================================================
COLLECTION_INTERVAL = 1  # Seconds between data collection polls
PREDICTION_INTERVAL = 5  # How often to predict the future

# NETWORK CONFIG
PT_IP = os.getenv('PT_IP', '192.168.2.4')
RYU_API_URL = f"http://{PT_IP}:8080"

# MODEL CONFIG
# We use the 'mixed' classifier because it was trained on ALL scenarios
MODEL_PATH = os.path.join(project_root, 'models', 'mixed_classifier_3050.pt')

# TOPOLOGY CONFIG
CURRENT_TOPO_NAME = "GENERATED_MESH_3"
REDUNDANCY_CONFIG = {
    "GENERATED_MESH_3": {
        1: 2,  # Switch 1 fails over to Port 2 (connecting to Switch 3)
    }
}


# =============================================================================
# COLLECTOR (Background Thread)
# =============================================================================
def collect_data_periodically():
    """
    Polls Ryu Controller and saves raw stats to SQLite.
    (Same as before - keeps the Digital Twin synchronized)
    """
    db = get_db()
    print(f"[Collector] STARTED: Polling every {COLLECTION_INTERVAL}s")

    while True:
        try:
            switches = get_active_switches()
            if not switches:
                time.sleep(COLLECTION_INTERVAL)
                continue

            for dpid in switches:
                # Collect Port Stats
                try:
                    url = f"{RYU_API_URL}/stats/port/{dpid}"
                    resp = requests.get(url, timeout=0.5)
                    if resp.status_code == 200:
                        ports = resp.json().get(str(dpid), [])
                        for port in ports:
                            db.save_port_stats(
                                dpid=dpid,
                                port_no=port.get('port_no', 0),
                                rx_packets=port.get('rx_packets', 0),
                                tx_packets=port.get('tx_packets', 0),
                                rx_bytes=port.get('rx_bytes', 0),
                                tx_bytes=port.get('tx_bytes', 0)
                            )
                except Exception:
                    pass

                # (Flow stats collection omitted for brevity, but keep it if you use it)

            # Save Hosts
            hosts = get_hosts()
            for host in hosts:
                db.save_host(host['mac'], host['dpid'], host['port'])

        except Exception as e:
            print(f"[Collector] ERROR: {e}")

        time.sleep(COLLECTION_INTERVAL)


def print_banner():
    print("\n" + "=" * 60)
    print("   DIGITAL TWIN ORCHESTRATOR (CLASSIFICATION MODE)   ")
    print("   Networking DT4SDN Project   ")
    print("=" * 60)
    print(f"Target Physical Twin: {RYU_API_URL}")
    print(f"AI Model: {os.path.basename(MODEL_PATH)}")
    print("=" * 60 + "\n")


# =============================================================================
# PREDICTOR (The Brain)
# =============================================================================
def run_prediction_loop():
    """
    Background thread to predict future traffic using the trained models.
    Uses Statistical Dispatching to select the correct specialist.
    """

    # --- 1. LOAD ALL SPECIALISTS ---
    print(f"[Predictor] Loading specialist models...")
    predictors = {}
    try:
        # Load all models defined in the MODELS config dictionary
        for scenario, paths in MODELS.items():
            model_path = paths['model']
            if os.path.exists(model_path):
                print(f"   ✓ Loading {scenario} specialist...")
                # StatePredictor handles scaling internally
                predictors[scenario] = StatePredictor(model_path)
            else:
                print(f"Warning: {scenario} model not found at {model_path}")

        if not predictors:
            print(" Error: No models loaded. Prediction loop will effectively do nothing.")
            return

    except Exception as e:
        print(f" [Predictor] Failed to load models: {e}")
        return

    # Database & Manager Access
    db = get_db()
    traffic_manager = TrafficManager()

    print("[Predictor] AI Models Active. Monitoring links...")

    while True:
        try:
            # 2. Get Active Links
            active_links = db.get_active_links()
            if not active_links:
                time.sleep(PREDICTION_INTERVAL)
                continue

            for link in active_links:
                dpid = link['dpid']
                port = link['port']
                link_id = f"s{dpid}-eth{port}"

                # Skip LOCAL ports (internal switch ports)
                if 'LOCAL' in str(port):
                    continue

                # 3. FETCH HISTORY (Watch 90 seconds)
                # We need exactly 90s to match the training window
                history_df = db.get_recent_traffic(link_id, duration_seconds=90)

                # Ensure we have enough data points
                # (Allow slightly less than 90 if starting up, e.g., > 85)
                if len(history_df) < 85:
                    # print(f"[Debug] {link_id}: Gathering history ({len(history_df)}/90)...")
                    continue

                traffic_data = history_df['bytes_sent'].values

                # 4. THE DISPATCHER (The Brain)
                # Analyze the statistics of the last 90 seconds to pick the expert
                detected_scenario = determine_scenario(traffic_data)

                # Debug log to see which expert is being chosen
                # print(f"🔎 [{link_id}] Analysis: {detected_scenario}")

                # 5. SELECT THE EXPERT MODEL
                # If we detected 'DDOS', try to use the 'DDOS' model.
                # If 'DDOS' model isn't loaded, fallback to 'MIXED', then 'NORMAL'.
                active_model = predictors.get(detected_scenario)

                if not active_model:
                    # Fallbacks if specific expert is missing
                    active_model = predictors.get('MIXED') or predictors.get('NORMAL')

                if not active_model:
                    print(f"Error: No suitable model found for {detected_scenario}")
                    continue

                # 6. PREDICT STATE (Future 60 seconds)
                try:
                    result = active_model.predict(traffic_data)

                    predicted_state = result['state']  # e.g., "CRITICAL", "HIGH"
                    confidence = result['confidence']  # e.g., 0.95
                    est_bandwidth = result['estimated_bandwidth']  # estimated bytes

                    # Sanity Log
                    print(f"🤖 [{link_id}] {detected_scenario} Specialist -> Pred: {predicted_state} ({confidence:.0%})")

                    # 7. STORE PREDICTION (For Dashboard)
                    db.store_prediction(
                        dpid=dpid,
                        port_no=port,
                        predicted_bytes=float(est_bandwidth),
                        timestamp=time.time()
                    )

                except Exception as e:
                    print(f"❌ Prediction logic failed for {link_id}: {e}")

        except Exception as e:
            print(f"⚠ [Predictor] Main Loop Error: {e}")

        time.sleep(PREDICTION_INTERVAL)
def determine_scenario(traffic_window):
    """
    Analyzes a 90-second traffic window (list or numpy array of bytes)
    and returns the most likely scenario ('NORMAL', 'BURST', 'DDOS', 'CONGESTION').
    """
    if len(traffic_window) == 0:
        return 'NORMAL'

    # 1. Calculate Basic Stats
    avg_load = np.mean(traffic_window)
    max_load = np.max(traffic_window)
    std_dev = np.std(traffic_window)

    # Calculate Slope (Trend)
    # We create an x-axis [0, 1, 2...] to measure if traffic is going UP or DOWN
    x = np.arange(len(traffic_window))
    slope, _, _, _, _ = linregress(x, traffic_window)

    # --- THRESHOLDS (Adjust these based on your bandwidth, e.g. 30Mbps) ---
    # 30 Mbps link capacity ~= 3,750,000 bytes/sec
    # Let's say "High Load" is > 70% (2.6 MB)
    CAPACITY = 30 * 1e6 / 8  # 3.75 MB
    HIGH_LOAD_THRESHOLD = CAPACITY * 0.7
    LOW_LOAD_THRESHOLD = CAPACITY * 0.3

    # --- DECISION LOGIC ---

    # Case 1: Is it basically empty?
    if avg_load < LOW_LOAD_THRESHOLD:
        # Check for Burst (Tiny average, but huge spike)
        if max_load > HIGH_LOAD_THRESHOLD:
            return 'BURST'
        return 'NORMAL'

    # Case 2: It is heavy. Is it Burst or DDoS?
    if max_load > HIGH_LOAD_THRESHOLD:
        # BURST Logic: High volatility (spiky) AND Mean is significantly lower than Max
        # If the Average is less than 60% of the Max, it was likely a short burst.
        if avg_load < (max_load * 0.6) or std_dev > (avg_load * 0.5):
            return 'BURST'

        # DDOS Logic: Sustained high load. Low volatility (consistent pressure).
        else:
            return 'DDOS'

    # Case 3: Medium load. Is it getting worse?
    # If we have a strong positive slope, it's Congestion building up.
    if slope > (CAPACITY * 0.01):  # Rising faster than 1% capacity per second
        return 'CONGESTION'

    # Fallback
    return 'NORMAL'

# --- MAIN ENTRY POINT ---
if __name__ == '__main__':
    print_banner()

    t_col = threading.Thread(target=collect_data_periodically, daemon=True)
    t_col.start()

    t_pred = threading.Thread(target=run_prediction_loop, daemon=True)
    t_pred.start()

    try:
        start_web_server(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down Orchestrator...")
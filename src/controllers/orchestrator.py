import os
import sys
import time
import threading
import requests
import numpy as np
from scipy.stats import linregress

# -- GETTING PATHS TO ACCESS CLASSES INSTANCES --
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)
project_root = os.path.dirname(src_dir)

from database.db_manager import get_db
from ml_models.state_predictor import StatePredictor
from web_interface.app import start_web_server, get_active_switches, get_hosts
from utils.traffic_manager import TrafficManager

# =============================================================================
# CONFIGURATION
# =============================================================================
COLLECTION_INTERVAL = 1  # Seconds between data collection polls
PREDICTION_INTERVAL = 5  # How often to predict the future

# NETWORK CONFIG
PT_IP = os.getenv('PT_IP', '192.168.2.4')
RYU_API_URL = f"http://{PT_IP}:8080"

# --- MODEL CONFIGURATION ---
MODELS = {
    'NORMAL': {
        'model': os.path.join(project_root, 'models', 'normal_classifier_3050.pt'),
        'scaler': os.path.join(project_root, 'models', 'normal_classifier_3050_scaler.pkl')
    },
    'CONGESTION': {
        'model': os.path.join(project_root, 'models', 'congestion_classifier_3050.pt'),
        'scaler': os.path.join(project_root, 'models', 'congestion_classifier_3050_scaler.pkl')
    },
    'DDOS': {
        'model': os.path.join(project_root, 'models', 'ddos_classifier_3050.pt'),
        'scaler': os.path.join(project_root, 'models', 'ddos_classifier_3050_scaler.pkl')
    },
    'BURST': {
        'model': os.path.join(project_root, 'models', 'burst_classifier_3050.pt'),
        'scaler': os.path.join(project_root, 'models', 'burst_classifier_3050_scaler.pkl')
    },
    'MIXED': {
        'model': os.path.join(project_root, 'models', 'mixed_classifier_3050.pt'),
        'scaler': os.path.join(project_root, 'models', 'mixed_classifier_3050_scaler.pkl')
    }
}

# TOPOLOGY CONFIG
CURRENT_TOPO_NAME = "GENERATED_MESH_3"
REDUNDANCY_CONFIG = {
    "GENERATED_MESH_3": {
        1: 2,  # Switch 1 fails over to Port 2 (connecting to Switch 3)
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def determine_scenario(traffic_window):
    """
    Analyzes traffic to decide which specialist model to consult.
    Prioritizes RECENT activity over long-term averages to catch attacks faster.
    """
    if len(traffic_window) == 0:
        return 'NORMAL'

    # 1. Stats
    avg_load = np.mean(traffic_window)
    max_load = np.max(traffic_window)
    std_dev = np.std(traffic_window)

    # Check the last 5 seconds (The "Now")
    recent_window = traffic_window[-5:] if len(traffic_window) > 5 else traffic_window
    recent_avg = np.mean(recent_window)

    # 2. Thresholds (30 Mbps Link)
    # 30 Mbps = 3.75 MB/s
    CAPACITY = 30 * 1e6 / 8

    # LOWERED THRESHOLD: Trigger 'High' logic at 50% capacity (15 Mbps)
    HIGH_LOAD_THRESHOLD = CAPACITY * 0.5
    CRITICAL_LOAD_THRESHOLD = CAPACITY * 0.8

    # --- DECISION LOGIC ---

    # Rule 1: Instant Criticality (DDoS / Heavy Burst)
    if recent_avg > CRITICAL_LOAD_THRESHOLD:
        # Low volatility = Sustained Attack = DDOS
        if std_dev < (avg_load * 0.4):
            return 'DDOS'

# 2. The "Sustained High" Check (TUNED)
        # Instead of 10s, we check the last 25 seconds.
        # Ideally, a Burst should finish within 25 seconds.
        check_duration = 25 
        
        if len(traffic_window) > check_duration:
            sustained_window = traffic_window[-check_duration:]
            min_sustained = np.min(sustained_window)
            
            # If the LOWEST point in the last 25s is STILL critical,
            # then it never dropped. It's a DDoS.
            if min_sustained > CRITICAL_LOAD_THRESHOLD:
                return 'DDOS'

        # If it hasn't been 25 seconds yet, or if it dipped recently, keep calling it BURST.
        return 'BURST'

    # Rule 2: High Traffic (Congestion or Light DDoS)
    if recent_avg > HIGH_LOAD_THRESHOLD:
        # Calculate slope on the whole window
        x = np.arange(len(traffic_window))
        if len(x) > 1:
            slope, _, _, _, _ = linregress(x, traffic_window)
        else:
            slope = 0

        # Positive slope = Congestion building up
        if slope > 0:
            return 'CONGESTION'
        return 'DDOS' 

    # Rule 3: Low Recent Traffic (Normal)
    return 'NORMAL'


def collect_data_periodically():
    """Polls Ryu Controller and saves raw stats to SQLite."""
    db = get_db()
    print(f"[Collector] STARTED: Polling every {COLLECTION_INTERVAL}s")

    while True:
        try:
            switches = get_active_switches()
            if not switches:
                time.sleep(COLLECTION_INTERVAL)
                continue

            for dpid in switches:
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
                print("No active links...")
                continue

            for link in active_links:
                dpid = link['dpid']
                port = link['port']
                link_id = f"s{dpid}-eth{port}"

                # Skip LOCAL ports
                if 'LOCAL' in str(port):
                    continue

                # 3. FETCH HISTORY (Watch 90 seconds)
                history_df = db.get_recent_traffic(link_id, duration_seconds=90)

                if len(history_df) < 85:
                    continue

                traffic_data = history_df['bytes_sent'].values

                # 4. THE DISPATCHER (The Brain)
                detected_scenario = determine_scenario(traffic_data)

                # 5. SELECT THE EXPERT MODEL
                active_model = predictors.get(detected_scenario)

                if not active_model:
                    active_model = predictors.get('MIXED') or predictors.get('NORMAL')

                if not active_model:
                    print(f"Error: No suitable model found for {detected_scenario}")
                    continue

                # 6. PREDICT STATE (Future 60 seconds)
                try:
                    result = active_model.predict(traffic_data)

                    predicted_state = result['state']
                    confidence = result['confidence']
                    est_bandwidth = result['estimated_bandwidth']

                    print(f"🤖 [{link_id}] {detected_scenario} Specialist -> Pred: {predicted_state} ({confidence:.0%})")

                    # 7. STORE PREDICTION
                    db.store_prediction(
                        dpid=dpid,
                        port_no=port,
                        predicted_bytes=float(est_bandwidth),
                        timestamp=time.time()
                    )
                    
                    # (Mitigation logic removed as requested)

                except Exception as e:
                    print(f"❌ Prediction logic failed for {link_id}: {e}")

        except Exception as e:
            print(f"⚠ [Predictor] Main Loop Error: {e}")

        time.sleep(PREDICTION_INTERVAL)


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

import os
import sys
import time
import threading
import requests
import numpy as np

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

project_root= os.path.dirname(src_dir)

from database.db_manager import get_db
from ml_models.traffic_predictor import TrafficPredictor
from web_interface.app import start_web_server, get_active_switches, get_hosts, RYU_API_URL

# --- CONFIGURATION ---
COLLECTION_INTERVAL = 2  # How often to read from Physical Twin
PREDICTION_INTERVAL = 5  # How often to predict the future
# 2. Define Paths using the absolute project root

MODELS = {
    'NORMAL': {
        'model': os.path.join(project_root, 'models', 'normal_ultimate.pt'),
        'scaler': os.path.join(project_root, 'models', 'normal_ultimate_scaler.pkl')
    },
    'CONGESTION': {
        'model': os.path.join(project_root, 'models', 'congestion_ultimate.pt'),
        'scaler': os.path.join(project_root, 'models', 'congestion_ultimate_scaler.pkl')
    },
    'DDOS': {
        'model': os.path.join(project_root, 'models', 'ddos_ultimate.pt'),
        'scaler': os.path.join(project_root, 'models', 'ddos_ultimate_scaler.pkl')
    },
    'BURST': {
        'model': os.path.join(project_root, 'models', 'burst_ultimate.pt'),
        'scaler': os.path.join(project_root, 'models', 'burst_ultimate_scaler.pkl')
    }
}

PREDICTION_INTERVAL = 5     # Keep this at 5 seconds
LINK_BW_MBPS = 10
COLLECTION_INTERVAL = 2

# Helper to convert Mbps -> Bytes per Prediction Interval
def mbps_to_bytes(mbps, interval):
    # (Mbps * 1,000,000 bits) / 8 bits_per_byte * interval_seconds
    return (mbps * 1_000_000 / 8) * interval

# Normal: < 6 Mbps
LIMIT_NORMAL = mbps_to_bytes(6, PREDICTION_INTERVAL)

# Congestion: 8 Mbps to 12 Mbps (Center is 10 Mbps)
LIMIT_CONGESTION_START = mbps_to_bytes(8, PREDICTION_INTERVAL)
LIMIT_CONGESTION_END = mbps_to_bytes(13, PREDICTION_INTERVAL)

# DDoS/Burst: > 15 Mbps
LIMIT_DDOS_START = mbps_to_bytes(14, PREDICTION_INTERVAL)
PT_IP = os.getenv('PT_IP', '192.168.2.4')
RYU_API_URL = f"http://{PT_IP}:8080"
def collect_data_periodically():
    """Background thread to collect and store traffic data from Ryu."""
    db = get_db()
    print(f"✅ [Collector] Started - polling every {COLLECTION_INTERVAL}s")

    while True:
        try:
            # Get active switches from Ryu (using helper from app.py)
            switches = get_active_switches()

            for dpid in switches:
                # A. Collect Port Stats
                try:
                    url = f"{RYU_API_URL}/stats/port/{dpid}"
                    resp = requests.get(url, timeout=2)
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

                # B. Collect Flow Stats (Optional, good for debugging)
                try:
                    url = f"{RYU_API_URL}/stats/flow/{dpid}"
                    resp = requests.get(url, timeout=2)
                    flows = resp.json().get(str(dpid), [])

                    for flow in flows:
                        db.save_flow_stats(
                            dpid=dpid,
                            priority=flow.get('priority', 0),
                            match_rules=flow.get('match', {}),
                            packet_count=flow.get('packet_count', 0),
                            byte_count=flow.get('byte_count', 0),
                            actions=flow.get('actions', [])
                        )
                except Exception:
                    pass

            # C. Save Hosts
            hosts = get_hosts()
            for host in hosts:
                db.save_host(host['mac'], host['dpid'], host['port'])

        except Exception as e:
            print(f"⚠ [Collector] Error: {e}")

        time.sleep(COLLECTION_INTERVAL)


def run_prediction_loop():
    """Background thread to predict future traffic using the trained model."""

    # --- 1. INITIALIZE PREDICTORS ---
    predictors = {}
    try:
        for scenario, paths in MODELS.items():
            if os.path.exists(paths['model']):
                print(f"   👉 Loading {scenario} specialist...")
                predictors[scenario] = TrafficPredictor(paths['model'], paths['scaler'])
            else:
                print(f"   ⚠ Warning: {scenario} model not found at {paths['model']}")
    except Exception as e:
        print(f"❌ Critical: Failed to load models: {e}")
        return

    # --- 2. START LOOP ---
    db = get_db()

    while True:
        try:
            # 2. Get Active Links
            active_links = db.get_active_links()

            if not active_links:
                # print("[Predictor] No active links...")
                pass

            for link in active_links:
                dpid = link['dpid']
                port = link['port']
                link_id = f"s{dpid}-eth{port}"

                # Skip LOCAL ports
                if 'LOCAL' in str(port):
                    continue

                # 3. Check Data
                history = db.get_recent_traffic(link_id, duration_seconds=60)

                if history.empty:
                    print(f"⚠ [Debug] {link_id}: Insufficient history.")
                    continue

                # 4. DETERMINE SCENARIO (Dispatcher Logic)
               # Calculate average bytes per step from history
                avg_bytes_per_step = history['bytes_sent'].tail(3).mean()

                # Approximate Bytes Per Second (assuming ~2s collection interval)
                # Ideally, calculate this dynamically based on timestamps, but this is a safe patch:
                estimated_bps = avg_bytes_per_step / COLLECTION_INTERVAL
                # Convert the Thresholds to Bytes Per Second for comparison
                # (LIMIT constants are currently based on 5 seconds)
                limit_ddos_bps = LIMIT_DDOS_START / PREDICTION_INTERVAL
                limit_cong_bps = LIMIT_CONGESTION_START / PREDICTION_INTERVAL

                # Default
                current_scenario = 'NORMAL'
                active_model = predictors.get('NORMAL')

                if estimated_bps > limit_ddos_bps:
                    current_scenario = 'DDOS'
                    active_model = predictors.get('DDOS')
                elif estimated_bps > limit_cong_bps:
                    current_scenario = 'CONGESTION'
                    active_model = predictors.get('CONGESTION')

                if active_model is None:
                    print("Error: No models available.")
                    continue


                print(f"Current scenario chosen: {current_scenario}")

                # 5. PREDICT
                try:
                    result = active_model.predict_next_frame(link_id, db)

                    if result and 'predictions' in result:
                        predicted_bytes = result['predictions'][0]

                        print(f"[{current_scenario}] {link_id} -> Pred: {predicted_bytes / 1e6:.2f} MB")

                        db.store_prediction(
                            dpid=dpid,
                            port_no=port,
                            predicted_bytes=float(predicted_bytes),
                            timestamp=time.time()
                        )
                except ValueError as ve:
                    print(f"❌ [Model Error] {current_scenario} model failed on {link_id}: {ve}")
                except Exception as e:
                    print(f"Prediction error on {link_id}: {e}")

        except Exception as e:
            print(f"⚠ [Predictor] Main Loop Error: {e}")

        time.sleep(PREDICTION_INTERVAL)

# --- MAIN ENTRY POINT ---
if __name__ == '__main__':
    print("========================================")
    print("   DIGITAL TWIN ORCHESTRATOR v2.1")
    print("========================================")
    print(f"Target Physical Twin: {RYU_API_URL}")

    # 1. Start Collector Thread (Daemon)
    t_col = threading.Thread(target=collect_data_periodically, daemon=True)
    t_col.start()

    # 2. Start Predictor Thread (Daemon)
    t_pred = threading.Thread(target=run_prediction_loop, daemon=True)
    t_pred.start()

    # 3. Start Web Server (Blocking - Keeps app alive)
    # This calls the Flask app defined in src/web_interface/app.py
    try:
        start_web_server(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down Orchestrator...")

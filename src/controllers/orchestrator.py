import os
import sys
import time
import threading
import requests
import numpy as np

import sys
import os


from src.database.db_manager import get_db
from src.ml_models.traffic_predictor import TrafficPredictor
from src.web_interface.app import start_web_server, get_active_switches, get_hosts, RYU_API_URL

# --- CONFIGURATION ---
COLLECTION_INTERVAL = 2  # How often to read from Physical Twin
PREDICTION_INTERVAL = 5  # How often to predict the future
MODEL_PATH = 'models/traffic_model.pt'
SCALER_PATH = 'models/scaler.pkl'


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
    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠ [Predictor] Model not found at {MODEL_PATH}. Prediction disabled.")
        return

    print(f"✅ [Predictor] Loading model from {MODEL_PATH}...")

    try:
        # 2. Initialize Predictor (Loads PyTorch model into memory)
        predictor = TrafficPredictor(MODEL_PATH, SCALER_PATH if os.path.exists(SCALER_PATH) else None)
        print("✅ [Predictor] Model loaded successfully.")
    except Exception as e:
        print(f"❌ [Predictor] Failed to load model: {e}")
        return
"""
    # Create a dedicated DB connection for this thread
    db = get_db()

    while True:
        try:
            # 3. Identify Links to Predict
            # We want to predict traffic for every active port we have seen recently.
            # You might need to add 'get_active_links()' to your db_manager,
            # or just query distinct dpid/port pairs from the port_stats table.
            active_links = db.get_active_links()  # Returns list of dicts: [{'dpid': 1, 'port': 1}, ...]

            for link in active_links:
                dpid = link['dpid']
                port = link['port']
                link_id = f"s{dpid}-eth{port}"  # Create a unique ID string

                try:
                    # 4. Make Prediction
                    # The predict_next_frame method (from your teammate's code)
                    # handles fetching history from DB automatically.
                    result = predictor.predict_next_frame(link_id, db)

                    # Result is a dict: {'link_id': ..., 'predictions': [...], ...}
                    predicted_bytes = result['predictions'][0]  # Take the immediate next step

                    # 5. Store Prediction in DB
                    db.store_prediction(
                        dpid=dpid,
                        port_no=port,
                        predicted_bytes=float(predicted_bytes),
                        timestamp=time.time()
                    )

                    # Optional: Print to console to see it working
                    # print(f"🔮 [ML] {link_id}: {predicted_bytes:.0f} bytes predicted")

                except ValueError:
                    # This happens if there isn't enough history (e.g. startup)
                    pass
                except Exception as e:
                    print(f"⚠ [Predictor] Failed for {link_id}: {e}")

        except Exception as e:
            print(f"⚠ [Predictor] Loop Error: {e}")

        time.sleep(PREDICTION_INTERVAL)
"""

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
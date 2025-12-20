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
MODEL_PATH = os.path.join(project_root, 'models', 'congestion_ultimate.pt')
SCALER_PATH = os.path.join(project_root, 'models', 'congestion_ultimate_scaler.pkl')

PREDICTION_INTERVAL = 5     # Keep this at 5 seconds
LINK_BW_MBPS = 10

# Normal: < 6 Mbps
LIMIT_NORMAL = mbps_to_bytes(6, PREDICTION_INTERVAL)

# Congestion: 8 Mbps to 12 Mbps (Center is 10 Mbps)
LIMIT_CONGESTION_START = mbps_to_bytes(8, PREDICTION_INTERVAL)
LIMIT_CONGESTION_END = mbps_to_bytes(13, PREDICTION_INTERVAL)

# DDoS/Burst: > 15 Mbps
LIMIT_DDOS_START = mbps_to_bytes(14, PREDICTION_INTERVAL)

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
        print(f"⚠ [Predictor] Model not found at {MODEL_PATH}")
        return

    print(f" [Predictor] Loading model from {MODEL_PATH}...")

    try:
        predictor = TrafficPredictor(MODEL_PATH, SCALER_PATH if os.path.exists(SCALER_PATH) else None)
        print("[Predictor] Model loaded successfully.")
    except Exception as e:
        print(f"[Predictor] Failed to load model: {e}")
        return

    db = get_db()

    while True:
        try:
            # 2. Get Active Links
            active_links = db.get_active_links()
            # Debug: Verify we actually see links
            if not active_links:
                print("[Predictor] No active links found in the last 5 minutes.")

            for link in active_links:
                dpid = link['dpid']
                port = link['port']
                link_id = f"s{dpid}-eth{port}"

                # Skip LOCAL ports (internal switch ports)
                if 'LOCAL' in str(port):
                    continue

                # 3. DEBUG: Check Data BEFORE Prediction
                # This reveals if the DB query is returning 0 rows due to timezone mismatch
                history = db.get_recent_traffic(link_id, duration_seconds=60)

                if history.empty:
                    print(f"⚠ [Debug] {link_id}: History is EMPTY. (Check Timezones!)")
                    continue

                if len(history) < 10:
                    print(f"⚠ [Debug] {link_id}: History too short ({len(history)} rows).")
                    continue

                # 4. Attempt Prediction
                try:
                    print(f"[Predictor] Predicting for {link_id} with {len(history)} rows...")
                    result = predictor.predict_next_frame(link_id, db)

                    if result and 'predictions' in result:
                        predicted_bytes = result['predictions'][0]
                        print(f"[Result] {link_id} -> {predicted_bytes:.2f}")

                        #Check status of the link among the various scenarios: Normale, Burst, DDoS, Congestion
                        status = "UNKNOWN"

                        if pred_bytes < LIMIT_CONGESTION_START:
                            status = "NORMAL"
                            print(f"{status} | {link_id}: {predicted_bytes / 1e6:.2f} MB (Safe)")

                        elif LIMIT_CONGESTION_START <= predicted_bytes < LIMIT_DDOS_START:
                            status = "⚠ CONGESTION"
                            print(f"{status} | {link_id}: {predicted_bytes / 1e6:.2f} MB -> Link Saturation likely!")
                            # Action: Reroute traffic?

                        elif predicted_bytes >= LIMIT_DDOS_START:
                            status = " !!! DDOS / BURST"
                            print(f"{status} | {link_id}: {predicted_bytes / 1e6:.2f} MB -> SEVERE OVERLOAD!")
                            # Action: Block port? Apply QoS?

                        db.store_prediction(
                            dpid=dpid,
                            port_no=port,
                            predicted_bytes=float(predicted_bytes),
                            timestamp=time.time()
                        )
                    else:
                        print(f"[Predictor] Model returned None for {link_id}")

                except ValueError as ve:
                    print(f"[Predictor] ValueError on {link_id}: {ve}")
                except Exception as e:
                    print(f"[Predictor] Crash on {link_id}: {e}")

        except Exception as e:
            print(f"⚠ [Predictor] Main Loop Error: {e}")

        time.sleep(PREDICTION_INTERVAL)

# Helper to convert Mbps -> Bytes per Prediction Interval
def mbps_to_bytes(mbps, interval):
    # (Mbps * 1,000,000 bits) / 8 bits_per_byte * interval_seconds
    return (mbps * 1_000_000 / 8) * interval


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

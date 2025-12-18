import os
import sys
import time
import threading
import requests

# 1. Setup paths to import modules from sibling directories
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from database.db_manager import get_db
from web_interface.app import start_web_server, get_active_switches, get_hosts, RYU_API_URL

# Try to import TrafficPredictor
try:
    from ml_models.traffic_predictor import TrafficPredictor
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ [Orchestrator] ML dependencies not found: {e}")
    ML_AVAILABLE = False

# --- CONFIGURATION ---
COLLECTION_INTERVAL = 1  
PREDICTION_INTERVAL = 5
MODEL_PATH = os.path.join(src_dir, 'ml_models', 'congestion_ultimate.pt')
SCALER_PATH = os.path.join(src_dir, 'ml_models', 'scaler.pkl')


def collect_data_periodically():
    """Background thread to collect and store traffic data from Ryu."""
    db = get_db()
    print(f"✅ [Collector] Started - polling every {COLLECTION_INTERVAL}s")

    while True:
        try:
            switches = get_active_switches()
            if not switches:
                # print("... waiting for switches ...")
                pass

            for dpid in switches:
                # Collect Port Stats
                try:
                    url = f"{RYU_API_URL}/stats/port/{dpid}"
                    resp = requests.get(url, timeout=0.5)
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

                # Collect Flow Stats
                try:
                    url = f"{RYU_API_URL}/stats/flow/{dpid}"
                    resp = requests.get(url, timeout=0.5)
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

            # Save Hosts
            hosts = get_hosts()
            for host in hosts:
                db.save_host(host['mac'], host['dpid'], host['port'])

        except Exception as e:
            print(f"⚠ [Collector] Error: {e}")

        time.sleep(COLLECTION_INTERVAL)


def run_prediction_loop():
    """Background thread to predict future traffic using the trained model."""
    if not ML_AVAILABLE:
        print("⚠ [Predictor] ML Module not available. Prediction disabled.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"⚠ [Predictor] Model not found at {MODEL_PATH}. Prediction disabled.")
        return

    print(f"✅ [Predictor] Loading model from {MODEL_PATH}...")

    try:
        scaler = SCALER_PATH if os.path.exists(SCALER_PATH) else None
        predictor = TrafficPredictor(MODEL_PATH, scaler)
        print("✅ [Predictor] Model loaded successfully.")
    except Exception as e:
        print(f"❌ [Predictor] Failed to load model: {e}")
        return

    db = get_db()

    while True:
        try:
            switches = get_active_switches()
            
            if not switches:
                print("⏳ [Predictor] No switches found yet...")
            
            for dpid in switches:
                try:
                    url = f"{RYU_API_URL}/stats/port/{dpid}"
                    resp = requests.get(url, timeout=1)
                    ports = resp.json().get(str(dpid), [])
                    
                    for port_info in ports:
                        port_no = port_info.get('port_no')
                        if port_no == 'LOCAL': continue 
                        
                        try:
                            # DEBUG: Print what we are trying to do
                            # print(f"🔍 [Predictor] Analyzing s{dpid}:p{port_no}...")

                            result = predictor.predict_next_frame(dpid, port_no, db)
                            
                            prediction = result['predictions']
                            if hasattr(prediction, 'item'):
                                predicted_val = prediction.item()
                            elif hasattr(prediction, '__iter__'):
                                predicted_val = float(prediction[0])
                            else:
                                predicted_val = float(prediction)
                            
                            db.save_prediction(
                                dpid=dpid,
                                predicted_packets=0,
                                predicted_bytes=int(predicted_val),
                                horizon=predictor.prediction_horizon
                            )
                            
                            print(f"🔮 [ML] s{dpid}:p{port_no} -> Predicted {predicted_val:.0f} bytes")
                            
                        except ValueError as ve:
                            # This is normal during startup (first 60s)
                            # Uncomment next line to see "Insufficient data" messages
                            # print(f"⏳ [Predictor] Waiting for data s{dpid}:p{port_no}: {ve}")
                            pass
                        except Exception as e:
                            print(f"⚠ [Predictor] Error on s{dpid}:p{port_no}: {e}")
                            
                except Exception as e:
                    print(f"⚠ [Predictor] Failed to fetch ports for s{dpid}: {e}")

        except Exception as e:
            print(f"⚠ [Predictor] Loop Error: {e}")

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

    # 3. Start Web Server (Blocking)
    try:
        print("✅ [Web] Starting Dashboard on port 5000...")
        start_web_server(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down Orchestrator...")

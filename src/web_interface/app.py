# src/web_interface/app.py
import os
import sys
import requests
import json
import time
import numpy as np
from flask import Flask, render_template, jsonify, request
from collections import defaultdict # Added for traffic calculation

app = Flask(__name__)

# --- FIX IMPORT ---
# Ensure 'src' directory is in python path to allow imports from sibling directories
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Now we can import directly from 'database' (since 'src' is in path)
from database.db_manager import get_db
from ml_models.state_predictor import StateManager
from ml_models.seq2seq_predictor import Seq2SeqManager

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_dir = os.path.join(project_root, 'models')

# Initialize model managers (support multiple scenarios)
state_manager = None
seq2seq_manager = None

try:
    state_manager = StateManager(models_dir)
    print(f"✓ StateManager initialized with models: {state_manager.get_available_models()}")
except Exception as e:
    print(f"⚠️ Could not initialize StateManager: {e}")

try:
    seq2seq_manager = Seq2SeqManager(models_dir)
    print(f"✓ Seq2SeqManager initialized with models: {seq2seq_manager.get_available_models()}")
except Exception as e:
    print(f"⚠️ Could not initialize Seq2SeqManager: {e}")

# Backwards compatibility aliases
state_predictor = state_manager
seq2seq_predictor = seq2seq_manager

# --- CONFIGURATION ---
PT_IP = os.getenv('PT_IP', '192.168.2.4')
RYU_API_URL = f"http://{PT_IP}:8080"


# --- HELPER FUNCTIONS ---
def get_active_switches():
    """Ask Ryu for list of connected switches."""
    try:
        url = f"{RYU_API_URL}/stats/switches"
        resp = requests.get(url, timeout=2)
        return resp.json()
    except:
        return []

def get_hosts():
    """Extract hosts from flow rules."""
    switches = get_active_switches()
    hosts = []
    seen_macs = set()

    for dpid in switches:
        try:
            url = f"{RYU_API_URL}/stats/flow/{dpid}"
            resp = requests.get(url, timeout=2)
            data = resp.json()
            flows = data.get(str(dpid), [])

            for flow in flows:
                if flow.get('priority', 0) > 0:
                    match = flow.get('match', {})
                    eth_src = match.get('eth_src') or match.get('dl_src')
                    in_port = match.get('in_port')

                    if eth_src and eth_src not in seen_macs:
                        seen_macs.add(eth_src)
                        hosts.append({'mac': eth_src, 'dpid': dpid, 'port': in_port})
        except:
            continue
    return hosts


def get_all_flow_stats():
    """Fetch flow stats for all active switches."""
    switches = get_active_switches()
    all_stats = {}

    if not switches:
        return None, "Disconnected (No Switches Found)"

    try:
        for dpid in switches:
            url = f"{RYU_API_URL}/stats/flow/{dpid}"
            resp = requests.get(url, timeout=2)
            data = resp.json()
            all_stats[dpid] = data.get(str(dpid), [])
        return all_stats, "Connected"
    except Exception as e:
        return {}, f"Error: {str(e)}"


def get_topology_info():
    """Fetch the high-level topology type from Ryu."""
    try:
        url = f"{RYU_API_URL}/topology/metadata"
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        pass
    
    return {"type": "Unknown", "switches": [], "links": []}


# --- FLASK ROUTES ---
@app.route('/')
def index():
    # Fetch data
    switches = get_active_switches()
    hosts = get_hosts()
    flows, status = get_all_flow_stats()
    topo_info = get_topology_info()

    # Active DPIDs set
    active_dpids_set = set()
    for s_name in topo_info.get('switches', []):
        try:
            dpid_val = int(s_name.replace('s', ''))
            active_dpids_set.add(dpid_val)
        except:
            pass

    # DB Stats & Predictions for Cards
    try:
        db = get_db()
        db_stats_data = db.get_db_stats()
        
        predictions = {}
        with db._get_connection() as conn:
            # Get latest predictions
            rows = conn.execute('''
                SELECT dpid, port_no, predicted_bytes, timestamp 
                FROM predictions p1
                WHERE id = (
                    SELECT MAX(id) 
                    FROM predictions p2 
                    WHERE p2.dpid = p1.dpid AND p2.port_no = p1.port_no
                )
                ORDER BY dpid, port_no
            ''').fetchall()
            
            # Dynamic Max Calculation
            current_max_kb = 0.0
            row_data = []
            
            for row in rows:
                if row['dpid'] in active_dpids_set:
                    kb_val = row['predicted_bytes'] / 1000.0
                    if kb_val > current_max_kb:
                        current_max_kb = kb_val
                    row_data.append((row, kb_val))
            
            reference_max = max(current_max_kb, 50.0) 

            for row, kb_val in row_data:
                key = f"s{row['dpid']}:eth{row['port_no']}"
                ratio = kb_val / reference_max
               
                if ratio < 0.4:
                    level = 'low'; status_text = 'Normal'
                elif ratio < 0.75:
                    level = 'medium'; status_text = 'Elevated'
                else:
                    level = 'high'; status_text = 'High Load'
               
                if kb_val < 10.0:
                    level = 'low'; status_text = 'Idle'

                predictions[key] = {
                    'dpid': row['dpid'],
                    'port': row['port_no'],
                    'value': f"{kb_val:.1f} KB",
                    'timestamp': row['timestamp'],
                    'level': level,
                    'status': status_text
                }
        
    except Exception as e:
        print(f"DB Error: {e}")
        db_stats_data = {'traffic_stats': 0, 'flow_stats': 0, 'hosts': 0, 'predictions': 0}
        predictions = {}

    return render_template(
        'index.html',
        pt_ip=PT_IP,
        switches=switches,
        hosts=hosts,
        flows=flows,
        connection_status=status,
        topo_info=topo_info,
        db_stats=db_stats_data,
        predictions=predictions
    )


# --- API ENDPOINTS FOR GRAPH & MODEL SELECTION ---

@app.route('/api/models', methods=['GET'])
def get_current_models():
    """Returns currently active models"""
    return jsonify({
        'classifier': {
            'current_model': state_manager.current_scenario,
            'available': state_manager.get_available_models()
        } if state_manager else None,
        'seq2seq': {
            'current_model': seq2seq_manager.current_scenario,
            'available': seq2seq_manager.get_available_models()
        } if seq2seq_manager else None
    })

@app.route('/api/models/switch', methods=['POST'])
def switch_model_scenario():
    """Switches both models to a specific scenario manually"""
    data = request.json
    scenario = data.get('scenario', 'mixed')
    
    print(f"🔄 Manual switch request to: {scenario}")
    
    # 1. Switch Classifier Model
    cl_result = False
    if state_manager:
        cl_result = state_manager.switch_model(scenario)
    
    # 2. Switch Seq2Seq Model
    s2s_result = False
    if seq2seq_manager:
        s2s_result = seq2seq_manager.switch_model(scenario) # Using correct method name switch_model

    return jsonify({
        'status': 'success', 
        'scenario': scenario,
        'classifier_switched': cl_result,
        'seq2seq_switched': s2s_result
    })

# --- DATA PREPARATION HELPER ---
def get_traffic_data_for_plot(minutes=2):
    """
    Fetches raw traffic history, calculates aggregated per-second speed 
    across all ports to represent 'Network Load'.
    """
    db = get_db()
    traffic_history = db.get_traffic_history(minutes=minutes)
    
    if not traffic_history or len(traffic_history) < 2:
        return [], 0, 0

    # Group by (dpid, port) to handle cumulative counters correctly
    port_data = defaultdict(lambda: defaultdict(int))
    
    for entry in traffic_history:
        dpid = entry.get('dpid', 0)
        port = entry.get('port_no', 0)
        ts = entry.get('timestamp', '')
        # Total load = Tx + Rx
        total_bytes = entry.get('tx_bytes', 0) + entry.get('rx_bytes', 0)
        port_data[(dpid, port)][ts] = total_bytes
    
    # Get all unique timestamps sorted
    all_timestamps = set()
    for port_key in port_data:
        all_timestamps.update(port_data[port_key].keys())
    sorted_timestamps = sorted(all_timestamps)
    
    bytes_per_second = []
    
    # Calculate Sum of Deltas per second
    for i in range(1, len(sorted_timestamps)):
        prev_ts = sorted_timestamps[i-1]
        curr_ts = sorted_timestamps[i]
        
        total_delta = 0
        for port_key in port_data:
            prev_bytes = port_data[port_key].get(prev_ts, 0)
            curr_bytes = port_data[port_key].get(curr_ts, 0)
            
            # Simple overflow protection: if curr < prev, ignore (reset)
            if prev_bytes > 0 and curr_bytes >= prev_bytes:
                delta = curr_bytes - prev_bytes
                total_delta += delta
        
        bytes_per_second.append(total_delta)
        
    if not bytes_per_second:
        return [0], 0, 0
        
    current_traffic = bytes_per_second[-1]
    avg_traffic = np.mean(bytes_per_second)
    
    return bytes_per_second, current_traffic, avg_traffic


@app.route('/api/prediction')
def get_prediction():
    """Returns the latest prediction for the graph using the specialized managers"""
    try:
        # 1. Get Cleaned Traffic Data
        traffic_series, current_traffic, avg_traffic = get_traffic_data_for_plot(minutes=2)
        
        # Prepare Fallback Defaults
        response = {
            'state': 'NORMAL',
            'confidence': 0.0,
            'color': '#2ecc71', # Green
            'current_traffic': current_traffic,
            'future_values': [],
            'prediction_horizon': 30,
            'timestamp': time.time()
        }

        if len(traffic_series) < 10:
            # Not enough data for models
            response['state'] = 'LOADING...'
            return jsonify(response)

        # 2. Run Classification (State Manager)
        if state_manager:
            try:
                # Convert to numpy array for model
                input_data = np.array(traffic_series, dtype=np.float32)
                result = state_manager.predict(input_data)
                
                if result:
                    response['state'] = result['state']
                    response['confidence'] = result['confidence']
                    response['color'] = result['color']
                    # Use the estimated bandwidth from classifier as a reference point?
                    # response['estimated_bandwidth'] = result['estimated_bandwidth']
            except Exception as e:
                print(f"[API] Classification failed: {e}")

        # 3. Run Regression (Seq2Seq Manager)
        if seq2seq_manager:
            try:
                # Seq2Seq expects similar input
                input_data = np.array(traffic_series, dtype=np.float32)
                # Ensure we select a generic link or aggregate for the visualization
                # Since get_traffic_data_for_plot returns aggregate network load,
                # passing it to seq2seq might result in scale issues if model trained on single link.
                # However, for visualization, the shape matters most.
                
                # Note: seq2seq_manager.predict usually expects a link_id to fetch from DB, 
                # OR raw data. Let's assume we pass raw data directly if the class supports it.
                # If your Seq2SeqManager.predict ONLY accepts link_id, we might need a workaround.
                # Assuming updated Seq2SeqManager accepts raw numpy array:
                future_preds = seq2seq_manager.predict(input_data)
                
                if future_preds is not None:
                    response['future_values'] = future_preds.tolist() if hasattr(future_preds, 'tolist') else future_preds
                    response['prediction_horizon'] = len(response['future_values'])
            except Exception as e:
                print(f"[API] Seq2Seq failed: {e}")

        return jsonify(response)
        
    except Exception as e:
        print(f"API Prediction Error: {e}")
        return jsonify({'error': str(e)})

def _get_state_color(state):
    colors = {
        'NORMAL': '#2ecc71',
        'BURST': '#f1c40f',
        'CONGESTION': '#e67e22',
        'DDOS': '#e74c3c'
    }
    return colors.get(state.upper(), '#95a5a6')

@app.route('/stats')
def db_stats():
    db = get_db()
    return json.dumps(db.get_db_stats(), indent=2)


def start_web_server(host='0.0.0.0', port=5000):
    app.run(host=host, port=port)

if __name__ == '__main__':
    start_web_server()

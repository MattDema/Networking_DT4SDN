# src/web_interface/app.py
import os
import sys
import requests
import json
import time
import numpy as np
from flask import Flask, render_template, jsonify, request

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
    # import traceback; traceback.print_exc()

try:
    seq2seq_manager = Seq2SeqManager(models_dir)
    print(f"✓ Seq2SeqManager initialized with models: {seq2seq_manager.get_available_models()}")
except Exception as e:
    print(f"⚠️ Could not initialize Seq2SeqManager: {e}")
    # import traceback; traceback.print_exc()

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
        cl_result = state_manager.load_model(scenario)
    
    # 2. Switch Seq2Seq Model
    s2s_result = False
    if seq2seq_manager:
        s2s_result = seq2seq_manager.load_model(scenario)
        if s2s_result:
            # Force context reset so predictions don't glitch
            if hasattr(seq2seq_manager, 'reset_context'):
                seq2seq_manager.reset_context()

    return jsonify({
        'status': 'success', 
        'scenario': scenario,
        'classifier_switched': cl_result,
        'seq2seq_switched': s2s_result
    })

@app.route('/api/prediction')
def get_prediction():
    """Returns the latest prediction for the graph"""
    try:
        # 1. Capture Live Traffic (from DB usually, simplified here)
        # In a real setup, you'd query the DB for the last second of traffic
        # Here we simulate or fetch from a global state if available
        # db = get_db() ...
        
        # 2. Classifier Prediction
        current_state = "UNKNOWN"
        confidence = 0.0
        
        if state_manager:
            # Assume we have some way to get live input vector 
            # For now, we mock or use cached state
            # pred = state_manager.predict(input_vector)
            current_state = state_manager.current_scenario
            confidence = 0.85 # Mock confidence

        # 3. Time Series Prediction (Seq2Seq)
        future_values = []
        current_traffic = 0
        
        # We need a reference link (e.g., s1-eth2)
        target_link_id = "s1-eth2" 
        
        if seq2seq_manager:
            db = get_db()
            result = seq2seq_manager.predict_next_window(target_link_id, db)
            if result:
                future_values = result['predictions']
                # Get the last actual value for alignment
                # current_traffic = ...
                
        # Mocking data if model returns nothing (for UI testing)
        if not future_values:
            future_values = [] 

        return jsonify({
            'state': current_state,
            'confidence': confidence,
            'color': _get_state_color(current_state),
            'prediction_horizon': 30, # seconds
            'current_traffic': current_traffic, # bytes/sec
            'future_values': future_values      # bytes/sec array
        })
        
    except Exception as e:
        print(f"API Error: {e}")
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
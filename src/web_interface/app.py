# src/web_interface/app.py
import os
import sys
import requests
import json
from flask import Flask, render_template

app = Flask(__name__)

# --- FIX IMPORT ---
# Ensure 'src' directory is in python path to allow imports from sibling directories
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Now we can import directly from 'database' (since 'src' is in path)
from database.db_manager import get_db

# --- CONFIGURATION ---
# We define these here because the Dashboard needs them to fetch live data
PT_IP = os.getenv('PT_IP', '127.0.0.1')
RYU_API_URL = f"http://{PT_IP}:8080"


# --- HELPER FUNCTIONS (Exposed for Orchestrator to use too) ---
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
    except:
        pass
    return {"type": "Unknown", "switches": [], "links": []}


# --- FLASK ROUTES ---
@app.route('/')
def index():
    # Use the helpers defined above
    switches = get_active_switches()
    hosts = get_hosts()
    flows, status = get_all_flow_stats()
    topo_info = get_topology_info() # <-- Fetch topology info

    try:
        db = get_db()
        db_stats_data = db.get_db_stats()
        
        # Get latest predictions per port
        predictions = {}
        with db._get_connection() as conn:
            # Get the latest prediction for each dpid/port pair
            rows = conn.execute('''
                SELECT dpid, port, predicted_bytes, timestamp 
                FROM predictions p1
                WHERE id = (
                    SELECT MAX(id) 
                    FROM predictions p2 
                    WHERE p2.dpid = p1.dpid AND p2.port = p1.port
                )
                ORDER BY dpid, port
            ''').fetchall()
            
            # 1. Calculate Dynamic Max (Peak traffic in this snapshot)
            current_max_kb = 0.0
            row_data = []
            
            for row in rows:
                kb_val = row['predicted_bytes'] / 1000.0
                if kb_val > current_max_kb:
                    current_max_kb = kb_val
                row_data.append((row, kb_val))
            
            # Set a baseline threshold to avoid "Red" alerts for tiny background noise
            # If the highest traffic is < 50KB, we consider the network "Idle" (Green)
            # Otherwise, we scale based on the peak.
            reference_max = max(current_max_kb, 50.0) 

            for row, kb_val in row_data:
                key = f"s{row['dpid']}:p{row['port']}"
                
                # Calculate ratio against the current peak
                ratio = kb_val / reference_max
                
                if ratio < 0.4:
                    level = 'low'
                    status_text = 'Normal'
                elif ratio < 0.75:
                    level = 'medium'
                    status_text = 'Elevated'
                else:
                    level = 'high'
                    status_text = 'High Load'
                
                # Override: If traffic is absolutely very low, force Green
                if kb_val < 10.0:
                    level = 'low'
                    status_text = 'Idle'

                predictions[key] = {
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
        topo_info=topo_info,  # <-- Pass to template
        db_stats=db_stats_data,
        predictions=predictions
    )


@app.route('/stats')
def db_stats():
    db = get_db()
    return json.dumps(db.get_db_stats(), indent=2)


def start_web_server(host='0.0.0.0', port=5000):
    app.run(host=host, port=port)
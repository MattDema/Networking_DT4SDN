# src/web_interface/app.py
import os
import sys
import requests
import json
from flask import Flask, render_template_string


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

app = Flask(__name__)

from database.db_manager import get_db

# - CONFIGURATION ---
# We define these here because the Dashboard needs them to fetch live data
PT_IP = os.getenv('PT_IP', '192.168.2.4')
RYU_API_URL = f"http://{PT_IP}:8080"

# --- HTML TEMPLATE ---
# (Pasting your exact HTML template here)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Digital Twin Orchestrator</title>
    <style>
        :root { --bg-primary: #f0f0f0; --bg-card: #ffffff; --text-primary: #333333; --accent: #007bff; }
        [data-theme="dark"] { --bg-primary: #1a1a2e; --bg-card: rgba(255,255,255,0.05); --text-primary: #eeeeee; --accent: #00d4ff; }
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background: var(--bg-primary); color: var(--text-primary); }
        .card { background: var(--bg-card); padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;}
        h1 { color: var(--accent); }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .status-connected { color: #28a745; } .status-disconnected { color: #dc3545; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .stat-value { font-size: 36px; font-weight: bold; }
    </style>
    <meta http-equiv="refresh" content="5">
</head>
<body data-theme="light">
    <h1>🌐 Digital Twin Orchestrator</h1>
    <div class="grid">
        <div class="card">
            <h3>📡 Connection</h3>
            <p>Target: <strong>{{ pt_ip }}</strong></p>
            <p>Status: <span class="status-{{ 'connected' if connection_status == 'Connected' else 'disconnected' }}">{{ connection_status }}</span></p>
        </div>
        <div class="card">
            <h3>🗄️ Database</h3>
            <p>Traffic Records: <strong>{{ db_stats.traffic_stats }}</strong></p>
            <p>Flow Snapshots: <strong>{{ db_stats.flow_stats }}</strong></p>
        </div>
    </div>

    <div class="card">
        <h3>📊 Live Flow Statistics</h3>
        {% if flows %}
            <table>
                <thead><tr><th>Switch</th><th>Priority</th><th>Packets</th><th>Bytes</th><th>Actions</th></tr></thead>
                <tbody>
                    {% for dpid, flow_list in flows.items() %}
                        {% for flow in flow_list %}
                        <tr>
                            <td>s{{ dpid }}</td>
                            <td>{{ flow.priority }}</td>
                            <td>{{ flow.packet_count }}</td>
                            <td>{{ flow.byte_count }}</td>
                            <td><code>{{ flow.actions }}</code></td>
                        </tr>
                        {% endfor %}
                    {% endfor %}
                </tbody>
            </table>
        {% else %}
            <p>No flow data available yet.</p>
        {% endif %}
    </div>
</body>
</html>
"""


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


# --- FLASK ROUTES ---
@app.route('/')
def index():
    # Use the helpers defined above
    switches = get_active_switches()
    hosts = get_hosts()
    flows, status = get_all_flow_stats()

    try:
        db = get_db()
        db_stats_data = db.get_db_stats()
    except:
        db_stats_data = {'traffic_stats': 0, 'flow_stats': 0, 'hosts': 0}

    return render_template_string(
        DASHBOARD_HTML,
        pt_ip=PT_IP,
        switches=switches,
        hosts=hosts,
        flows=flows,
        connection_status=status,
        db_stats=db_stats_data
    )


@app.route('/stats')
def db_stats():
    db = get_db()
    return json.dumps(db.get_db_stats(), indent=2)


def start_web_server(host='0.0.0.0', port=5000):
    app.run(host=host, port=port)

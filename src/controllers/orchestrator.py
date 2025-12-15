# dummy_orchestrator.py
import os
import requests
import json
import time
from flask import Flask, render_template_string

app = Flask(__name__)

# --- CONFIGURATION ---
# Get Physical Twin IP from environment or use default
PT_IP = os.getenv('PT_IP', '127.0.0.1')
RYU_API_URL = f"http://{PT_IP}:8080"

# --- HTML TEMPLATE (Embedded for simplicity) ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Digital Twin Orchestrator</title>
    <style>
        body { font-family: sans-serif; padding: 20px; background: #f0f0f0; }
        .card { background: white; padding: 15px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; border-bottom: 1px solid #ddd; text-align: left; }
        th { background-color: #007bff; color: white; }
        .status { font-weight: bold; color: green; }
        .error { color: red; }
        .device-list { display: flex; gap: 30px; margin-top: 15px; }
        .device-section { flex: 1; }
        .device-section h4 { margin: 0 0 10px 0; color: #555; }
        .device-item { 
            display: inline-block; 
            background: #e7f3ff; 
            padding: 5px 12px; 
            margin: 3px; 
            border-radius: 15px; 
            font-size: 14px;
            border: 1px solid #007bff;
        }
        .device-item.switch { background: #fff3cd; border-color: #ffc107; }
        .device-item.host { background: #d4edda; border-color: #28a745; }
        .count-badge {
            background: #007bff;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
            margin-left: 5px;
        }
    </style>
    <meta http-equiv="refresh" content="5">
</head>
<body>
    <h1>Digital Twin Orchestrator</h1>

    <div class="card">
        <h3>Connection Status</h3>
        <p>Target Physical Twin: <strong>{{ pt_ip }}</strong></p>
        <p>Status: <span class="{{ status_class }}">{{ connection_status }}</span></p>
        
        <div class="device-list">
            <div class="device-section">
                <h4>🔀 Switches <span class="count-badge">{{ switches|length }}</span></h4>
                {% if switches %}
                    {% for sw in switches %}
                        <span class="device-item switch">s{{ sw }}</span>
                    {% endfor %}
                {% else %}
                    <p style="color: #999;">No switches detected</p>
                {% endif %}
            </div>
            
            <div class="device-section">
                <h4>💻 Hosts <span class="count-badge">{{ hosts|length }}</span></h4>
                {% if hosts %}
                    {% for host in hosts %}
                        <span class="device-item host">{{ host.mac }} (sw{{ host.dpid }}:p{{ host.port }})</span>
                    {% endfor %}
                {% else %}
                    <p style="color: #999;">No hosts detected yet. Generate traffic (ping) to discover hosts.</p>
                {% endif %}
            </div>
        </div>
    </div>

    <div class="card">
        <h3>Live Traffic Data (Switch Flow Stats)</h3>
        {% if flows %}
            <table>
                <thead>
                    <tr>
                        <th>Switch ID</th>
                        <th>Priority</th>
                        <th>Match Rules</th>
                        <th>Packets</th>
                        <th>Bytes</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for dpid, flow_list in flows.items() %}
                        {% for flow in flow_list %}
                        <tr>
                            <td>{{ dpid }}</td>
                            <td>{{ flow.priority }}</td>
                            <td>{{ flow.match }}</td>
                            <td><strong>{{ flow.packet_count }}</strong></td>
                            <td>{{ flow.byte_count }}</td>
                            <td>{{ flow.actions }}</td>
                        </tr>
                        {% endfor %}
                    {% endfor %}
                </tbody>
            </table>
        {% else %}
            <p>No flow data available yet. Generate traffic in Mininet (ping/iperf).</p>
        {% endif %}
    </div>
</body>
</html>
"""


# --- BACKEND LOGIC ---
def get_active_switches():
    """Ask Ryu for list of connected switches."""
    try:
        url = f"{RYU_API_URL}/stats/switches"
        resp = requests.get(url, timeout=2)
        return resp.json()  # Returns list like [1, 2, 3]
    except:
        return []


def get_hosts():
    """Extract hosts from flow rules (learned MAC addresses)."""
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
                # Only look at learned flows (priority > 0)
                if flow.get('priority', 0) > 0:
                    match = flow.get('match', {})
                    eth_src = match.get('eth_src') or match.get('dl_src')
                    in_port = match.get('in_port')
                    
                    if eth_src and eth_src not in seen_macs:
                        seen_macs.add(eth_src)
                        hosts.append({
                            'mac': eth_src,
                            'dpid': dpid,
                            'port': in_port
                        })
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
            # Clean up the key to be just the ID
            all_stats[dpid] = data.get(str(dpid), [])
        return all_stats, "Connected"
    except Exception as e:
        return {}, f"Error: {str(e)}"


# --- FLASK ROUTES ---
@app.route('/')
def index():
    switches = get_active_switches()
    hosts = get_hosts()
    flows, status = get_all_flow_stats()

    status_class = "status" if status == "Connected" else "error"

    return render_template_string(
        DASHBOARD_HTML,
        pt_ip=PT_IP,
        switches=switches,
        hosts=hosts,
        flows=flows,
        connection_status=status,
        status_class=status_class
    )


if __name__ == '__main__':
    # Run Web Server on port 5000, visible to host
    print(f"--- Starting Orchestrator ---")
    print(f"--- Target Physical Twin: {RYU_API_URL} ---")
    app.run(host='0.0.0.0', port=5000)
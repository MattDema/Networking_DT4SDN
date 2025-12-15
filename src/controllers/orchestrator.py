# dummy_orchestrator.py
import os
import requests
import json
import time
import threading
from flask import Flask, render_template_string

# Import database manager
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database.db_manager import get_db

app = Flask(__name__)

# --- CONFIGURATION ---
# Get Physical Twin IP from environment or use default
PT_IP = os.getenv('PT_IP', '127.0.0.1')
RYU_API_URL = f"http://{PT_IP}:8080"
COLLECTION_INTERVAL = 2  # seconds

# --- HTML TEMPLATE (Embedded for simplicity) ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Digital Twin Orchestrator</title>
    <style>
        :root {
            --bg-primary: #f0f0f0;
            --bg-card: #ffffff;
            --text-primary: #333333;
            --text-secondary: #666666;
            --accent: #007bff;
            --border: #dddddd;
            --switch-bg: #fff3cd;
            --switch-border: #ffc107;
            --host-bg: #d4edda;
            --host-border: #28a745;
            --table-header: #007bff;
            --table-hover: #f5f5f5;
            --shadow: rgba(0,0,0,0.1);
        }

        [data-theme="dark"] {
            --bg-primary: #1a1a2e;
            --bg-card: rgba(255,255,255,0.05);
            --text-primary: #eeeeee;
            --text-secondary: #888888;
            --accent: #00d4ff;
            --border: rgba(255,255,255,0.1);
            --switch-bg: rgba(255,193,7,0.2);
            --switch-border: #ffc107;
            --host-bg: rgba(40,167,69,0.2);
            --host-border: #28a745;
            --table-header: rgba(0,212,255,0.2);
            --table-hover: rgba(255,255,255,0.05);
            --shadow: rgba(0,0,0,0.3);
        }

        * { box-sizing: border-box; transition: background-color 0.3s, color 0.3s; }
        
        body { 
            font-family: 'Segoe UI', sans-serif; 
            padding: 20px; 
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            margin: 0;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        h1 { 
            color: var(--accent); 
            margin: 0;
        }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .live-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #0f0;
            font-size: 14px;
        }

        .live-dot {
            width: 10px;
            height: 10px;
            background: #0f0;
            border-radius: 50%;
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Theme Toggle Button */
        .theme-toggle {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 8px 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: var(--text-primary);
        }

        .theme-toggle:hover {
            border-color: var(--accent);
        }

        .theme-icon {
            font-size: 18px;
        }

        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
            gap: 20px; 
            margin-bottom: 20px;
        }

        .card { 
            background: var(--bg-card); 
            padding: 20px; 
            border-radius: 12px; 
            border: 1px solid var(--border);
            box-shadow: 0 2px 8px var(--shadow);
        }

        .card h3 { 
            color: var(--accent); 
            margin: 0 0 15px 0; 
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stat-value {
            font-size: 36px;
            font-weight: bold;
            color: var(--text-primary);
        }

        .stat-label {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 5px;
        }

        .status-connected { color: #28a745; font-weight: bold; }
        .status-disconnected { color: #dc3545; font-weight: bold; }

        .db-stats {
            display: flex;
            gap: 15px;
            margin-top: 10px;
            flex-wrap: wrap;
        }

        .db-stat {
            text-align: center;
            padding: 10px 15px;
            background: var(--bg-primary);
            border-radius: 8px;
            border: 1px solid var(--border);
            min-width: 70px;
        }

        .db-stat-value {
            font-size: 20px;
            font-weight: bold;
            color: var(--accent);
        }

        .db-stat-label {
            font-size: 10px;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .full-width { grid-column: 1 / -1; }

        .device-list { display: flex; gap: 20px; flex-wrap: wrap; }
        .device-section { flex: 1; min-width: 200px; }
        .device-section h4 { 
            margin: 0 0 10px 0; 
            color: var(--text-secondary); 
            font-size: 12px;
            text-transform: uppercase;
        }

        .device-item { 
            display: inline-block; 
            padding: 6px 14px; 
            margin: 3px; 
            border-radius: 20px; 
            font-size: 13px;
        }

        .device-item.switch { 
            background: var(--switch-bg); 
            border: 1px solid var(--switch-border); 
            color: var(--switch-border);
        }

        .device-item.host { 
            background: var(--host-bg); 
            border: 1px solid var(--host-border); 
            color: var(--host-border);
        }

        .count-badge {
            background: var(--accent);
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            margin-left: 8px;
            font-weight: bold;
        }

        [data-theme="dark"] .count-badge {
            color: #000;
        }

        table { width: 100%; border-collapse: collapse; }
        
        th, td { 
            padding: 12px; 
            text-align: left; 
            border-bottom: 1px solid var(--border); 
        }
        
        th { 
            background: var(--table-header); 
            color: var(--accent);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }

        [data-theme="light"] th {
            color: white;
        }
        
        td { 
            font-size: 13px; 
            color: var(--text-primary);
        }
        
        tr:hover { background: var(--table-hover); }

        code {
            background: var(--bg-primary);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
        }

        .no-data { 
            color: var(--text-secondary); 
            font-style: italic; 
            padding: 20px;
            text-align: center;
        }
    </style>
</head>
<body data-theme="light">
    <div class="header">
        <h1>🌐 Digital Twin Orchestrator</h1>
        <div class="header-controls">
            <div class="live-indicator">
                <span class="live-dot"></span>
                <span>LIVE</span>
            </div>
            <button class="theme-toggle" onclick="toggleTheme()">
                <span class="theme-icon" id="theme-icon">🌙</span>
                <span id="theme-text">Dark</span>
            </button>
        </div>
    </div>

    <div class="grid">
        <!-- Connection Status -->
        <div class="card">
            <h3>📡 Connection</h3>
            <p>Physical Twin: <strong>{{ pt_ip }}</strong></p>
            <p>Status: <span class="status-{{ 'connected' if connection_status == 'Connected' else 'disconnected' }}">
                {{ connection_status }}
            </span></p>
        </div>

        <!-- Switch Count -->
        <div class="card">
            <h3>🔀 Switches</h3>
            <div class="stat-value">{{ switches|length }}</div>
            <div class="stat-label">Active OpenFlow Switches</div>
        </div>

        <!-- Host Count -->
        <div class="card">
            <h3>💻 Hosts</h3>
            <div class="stat-value">{{ hosts|length }}</div>
            <div class="stat-label">Discovered Hosts</div>
        </div>

        <!-- Database Stats -->
        <div class="card">
            <h3>🗄️ Database</h3>
            <div class="db-stats">
                <div class="db-stat">
                    <div class="db-stat-value">{{ db_stats.traffic_stats }}</div>
                    <div class="db-stat-label">Traffic</div>
                </div>
                <div class="db-stat">
                    <div class="db-stat-value">{{ db_stats.flow_stats }}</div>
                    <div class="db-stat-label">Flows</div>
                </div>
                <div class="db-stat">
                    <div class="db-stat-value">{{ db_stats.hosts }}</div>
                    <div class="db-stat-label">Hosts</div>
                </div>
            </div>
        </div>

        <!-- Network Topology -->
        <div class="card full-width">
            <h3>🖧 Network Topology</h3>
            <div class="device-list">
                <div class="device-section">
                    <h4>Switches <span class="count-badge">{{ switches|length }}</span></h4>
                    {% if switches %}
                        {% for sw in switches %}
                            <span class="device-item switch">s{{ sw }}</span>
                        {% endfor %}
                    {% else %}
                        <span class="no-data">No switches detected</span>
                    {% endif %}
                </div>
                
                <div class="device-section">
                    <h4>Hosts <span class="count-badge">{{ hosts|length }}</span></h4>
                    {% if hosts %}
                        {% for host in hosts %}
                            <span class="device-item host">{{ host.mac }} → s{{ host.dpid }}:p{{ host.port }}</span>
                        {% endfor %}
                    {% else %}
                        <span class="no-data">Generate traffic (ping) to discover hosts</span>
                    {% endif %}
                </div>
            </div>
        </div>

        <!-- Flow Table -->
        <div class="card full-width">
            <h3>📊 Live Flow Statistics</h3>
            {% if flows %}
                <table>
                    <thead>
                        <tr>
                            <th>Switch</th>
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
                                <td>s{{ dpid }}</td>
                                <td>{{ flow.priority }}</td>
                                <td><code>{{ flow.match }}</code></td>
                                <td><strong>{{ flow.packet_count }}</strong></td>
                                <td>{{ flow.byte_count }}</td>
                                <td><code>{{ flow.actions }}</code></td>
                            </tr>
                            {% endfor %}
                        {% endfor %}
                    </tbody>
                </table>
            {% else %}
                <p class="no-data">No flow data available yet. Generate traffic in Mininet (ping/iperf).</p>
            {% endif %}
        </div>
    </div>

    <script>
        // Load saved theme preference
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.body.setAttribute('data-theme', savedTheme);
        updateThemeButton(savedTheme);

        function toggleTheme() {
            const body = document.body;
            const currentTheme = body.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            
            body.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeButton(newTheme);
        }

        function updateThemeButton(theme) {
            const icon = document.getElementById('theme-icon');
            const text = document.getElementById('theme-text');
            
            if (theme === 'dark') {
                icon.textContent = '☀️';
                text.textContent = 'Light';
            } else {
                icon.textContent = '🌙';
                text.textContent = 'Dark';
            }
        }
    </script>
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

    # Get DB stats
    try:
        db = get_db()
        db_stats_data = db.get_db_stats()
    except:
        db_stats_data = {'traffic_stats': 0, 'flow_stats': 0, 'hosts': 0, 'predictions': 0}

    return render_template_string(
        DASHBOARD_HTML,
        pt_ip=PT_IP,
        switches=switches,
        hosts=hosts,
        flows=flows,
        connection_status=status,
        db_stats=db_stats_data
    )


# --- DATA COLLECTION THREAD ---
def collect_data_periodically():
    """Background thread to collect and store traffic data."""
    db = get_db()
    print(f"[Collector] Started - saving data every {COLLECTION_INTERVAL}s")
    
    while True:
        try:
            switches = get_active_switches()
            
            for dpid in switches:
                # Collect port stats
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
                except Exception as e:
                    pass
                
                # Collect flow stats
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
                except Exception as e:
                    pass
            
            # Save discovered hosts
            hosts = get_hosts()
            for host in hosts:
                db.save_host(host['mac'], host['dpid'], host['port'])
                
        except Exception as e:
            print(f"[Collector] Error: {e}")
        
        time.sleep(COLLECTION_INTERVAL)


# Add new route to see DB stats
@app.route('/stats')
def db_stats():
    db = get_db()
    stats = db.get_db_stats()
    return json.dumps(stats, indent=2)


if __name__ == '__main__':
    # Start data collection thread
    collector_thread = threading.Thread(target=collect_data_periodically, daemon=True)
    collector_thread.start()
    
    print(f"--- Starting Orchestrator ---")
    print(f"--- Target Physical Twin: {RYU_API_URL} ---")
    app.run(host='0.0.0.0', port=5000)
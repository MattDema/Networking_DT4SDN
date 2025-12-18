# src/web_interface/app.py
import os
import sys
import requests
import json
from flask import Flask, render_template_string

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

# --- HTML TEMPLATE ---
# (Pasting your exact HTML template here)
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

        /* --- NEW TRAFFIC CARD STYLES --- */
        .traffic-grid {
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); 
            gap: 12px;
        }

        .traffic-card {
            padding: 12px; 
            border-radius: 8px; 
            text-align: center;
            border: 1px solid transparent;
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .traffic-low {
            background-color: rgba(40, 167, 69, 0.1);
            border-color: #28a745;
            color: #155724;
        }
        
        .traffic-medium {
            background-color: rgba(255, 193, 7, 0.15);
            border-color: #ffc107;
            color: #856404;
        }
        
        .traffic-high {
            background-color: rgba(220, 53, 69, 0.15);
            border-color: #dc3545;
            color: #721c24;
            animation: pulse-red 2s infinite;
            font-weight: bold;
        }

        @keyframes pulse-red {
            0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.4); }
            70% { box-shadow: 0 0 0 6px rgba(220, 53, 69, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
        }

        .traffic-key {
            font-size: 12px; 
            opacity: 0.8; 
            margin-bottom: 6px;
            font-family: monospace;
        }

        .traffic-value {
            font-size: 1.4em;
            font-weight: bold;
        }

        .traffic-status {
            font-size: 10px; 
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Dark mode adjustments for traffic cards */
        [data-theme="dark"] .traffic-low {
            background-color: rgba(40, 167, 69, 0.2);
            color: #75b798;
        }
        [data-theme="dark"] .traffic-medium {
            background-color: rgba(255, 193, 7, 0.2);
            color: #ffda6a;
        }
        [data-theme="dark"] .traffic-high {
            background-color: rgba(220, 53, 69, 0.3);
            color: #ea868f;
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

        <!-- AI Prediction -->
        <div class="card">
            <h3>🔮 AI Link Forecast (Next 30s)</h3>
            {% if predictions %}
                <div class="traffic-grid">
                    {% for key, data in predictions.items() %}
                    <div class="traffic-card traffic-{{ data.level }}">
                        <div class="traffic-key">{{ key }}</div>
                        <div class="traffic-value">{{ data.value }}</div>
                        <div class="traffic-status">{{ data.status }}</div>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="stat-label">Waiting for traffic data...</div>
            {% endif %}
        </div>

        <!-- Database -->
        <div class="card">
            <h3>🗄️ Database
                <span
                    class="info-inline"
                    title="Traffic: righe in 'traffic_stats' (snapshot porte). Flows: righe in 'flow_stats' (snapshot flow entries). Hosts: righe in 'hosts' (MAC unici)."
                    onclick="alert('Traffic: numero di snapshot delle statistiche delle porte salvati in \\ntraffic_stats (una riga per porta per ogni raccolta).\\n\\nFlows: snapshot delle flow entries salvate in \\nflow_stats (una riga per ogni flow per ogni raccolta).\\n\\nHosts: host scoperti salvati in \\nhosts (un record per MAC, aggiornato su ogni discovery).')"
                    style="cursor: pointer; margin-left: 8px; font-size: 14px; color: var(--text-secondary); padding: 2px 6px; border-radius: 12px; border: 1px solid var(--border); background: rgba(0,0,0,0.03);"
                >ⓘ</span>
            </h3>
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
        
        // Auto-refresh every 3 seconds
        setTimeout(function() {
            window.location.reload();
        }, 3000);
    </script>
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
        db_stats_data = {'traffic_stats': 0, 'flow_stats': 0, 'hosts': 0}
        predictions = {}

    return render_template_string(
        DASHBOARD_HTML,
        pt_ip=PT_IP,
        switches=switches,
        hosts=hosts,
        flows=flows,
        connection_status=status,
        db_stats=db_stats_data,
        predictions=predictions  # Pass the dict instead of single stats
    )


@app.route('/stats')
def db_stats():
    db = get_db()
    return json.dumps(db.get_db_stats(), indent=2)


def start_web_server(host='0.0.0.0', port=5000):
    app.run(host=host, port=port)
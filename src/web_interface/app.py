# src/web_interface/app.py
import os
import sys
import requests
import json
from flask import Flask, render_template_string, jsonify
import time

app = Flask(__name__)

# --- FIX IMPORT ---
# Ensure 'src' directory is in python path to allow imports from sibling directories
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Now we can import directly from 'database' (since 'src' is in path)
from database.db_manager import get_db
from ml_models.state_predictor import StatePredictor
import numpy as np

# Initialize state predictor (loaded once at startup)
state_predictor = None
try:
    import glob
    # Get project root (two levels up from this file: web_interface -> src -> project)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, 'models')
    
    model_pattern = os.path.join(models_dir, '*_classifier_3050.pt')
    model_files = glob.glob(model_pattern)
    
    print(f"Looking for models in: {models_dir}")
    print(f"Found models: {model_files}")
    
    if model_files:
        # Use mixed model as default (trained on all patterns)
        mixed_model = [f for f in model_files if 'mixed' in f]
        model_path = mixed_model[0] if mixed_model else model_files[0]
        state_predictor = StatePredictor(model_path)
        print(f"✓ StatePredictor loaded: {model_path}")
    else:
        print(f"⚠️ No classifier models found in {models_dir}")
except Exception as e:
    print(f"⚠️ Could not load StatePredictor: {e}")
    import traceback
    traceback.print_exc()

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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            grid-template-columns: 1fr; 
            gap: 20px; 
            margin-bottom: 20px;
        }

        /* New Layout Classes */
        .main-layout {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .main-content {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .top-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        @media (max-width: 1100px) {
            .main-layout {
                grid-template-columns: 1fr;
            }
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

    <div class="main-layout">
        <!-- Left Column: Database (1/3) -->
        <div class="sidebar">
            <div class="card" style="height: 100%;">
                <h3>🗄️ Database Storage</h3>
                <p style="font-size: 0.85em; color: var(--text-secondary); margin-bottom: 12px;">
                    SQLite storage for historical data and real-time monitoring.
                </p>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.85em;">
                    <thead>
                        <tr>
                            <th style="padding: 8px;">Table</th>
                            <th style="padding: 8px;">Entries</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="padding: 8px;"><code>traffic_stats</code></td>
                            <td style="padding: 8px;"><strong>{{ db_stats.traffic_stats }}</strong></td>
                        </tr>
                        <tr>
                            <td style="padding: 8px;"><code>flow_stats</code></td>
                            <td style="padding: 8px;"><strong>{{ db_stats.flow_stats }}</strong></td>
                        </tr>
                        <tr>
                            <td style="padding: 8px;"><code>hosts</code></td>
                            <td style="padding: 8px;"><strong>{{ db_stats.hosts }}</strong></td>
                        </tr>
                        <tr>
                            <td style="padding: 8px;"><code>predictions</code></td>
                            <td style="padding: 8px;"><strong>{{ db_stats.predictions }}</strong></td>
                        </tr>
                    </tbody>
                </table>
                <div style="margin-top: 20px; font-size: 0.8em; color: var(--text-secondary); line-height: 1.4;">
                    <p>• <b>Traffic:</b> Port metrics used for ML training.</p>
                    <p>• <b>Flows:</b> Snapshots of active OpenFlow rules.</p>
                    <p>• <b>Hosts:</b> Inventory of discovered devices.</p>
                    <p>• <b>Predictions:</b> ML-generated traffic forecasts.</p>
                </div>
            </div>
        </div>

        <!-- Right Column: Connection, Switches, Forecast (2/3) -->
        <div class="main-content">
            <div class="top-row">
                <!-- Connection Status -->
                <div class="card">
                    <h3>📡 Connection</h3>
                    <p style="margin: 5px 0; font-size: 0.9em;">PT: <strong>{{ pt_ip }}</strong></p>
                    <p style="margin: 5px 0; font-size: 0.9em;">Status: <span class="status-{{ 'connected' if connection_status == 'Connected' else 'disconnected' }}">
                        {{ connection_status }}
                    </span></p>
                </div>

                <!-- Switch Count -->
                <div class="card">
                    <h3>🔀 Switches</h3>
                    <div style="display: flex; align-items: baseline; gap: 10px;">
                        <div class="stat-value" style="font-size: 28px;">{{ switches|length }}</div>
                        <div class="stat-label">Active Nodes</div>
                    </div>
                </div>
            </div>

            <!-- AI Prediction -->
            <div class="card" style="flex-grow: 1;">
                <h3>🔮 AI Link Forecast (Next 30s)</h3>
                {% if predictions %}
                    <div class="traffic-grid">
                        {% for key, data in predictions.items() %}
                        <div class="traffic-card traffic-{{ data.level }}">
                            <div class="traffic-key">{{ key }}</div>
                            <div class="traffic-value" style="font-size: 1.2em;">{{ data.value }}</div>
                            <div class="traffic-status">{{ data.status }}</div>
                        </div>
                        {% endfor %}
                    </div>
                {% else %}
                    <div class="stat-label" style="padding: 20px; text-align: center;">Waiting for traffic data to generate forecasts...</div>
                {% endif %}
            </div>
        </div>
    </div>

    <!-- Full Width Sections -->
    <div class="grid">
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

        <!-- 📈 REAL-TIME PREDICTION GRAPH -->
        <div class="card full-width">
            <h3>📈 Traffic Prediction Graph (Physical vs Digital Twin)</h3>
            <div style="position: relative; height: 350px; width: 100%;">
                <canvas id="predictionChart"></canvas>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px; padding: 10px; background: var(--bg-primary); border-radius: 8px;">
                <div>
                    <span style="display: inline-block; width: 20px; height: 3px; background: #007bff; margin-right: 5px;"></span>
                    <span style="font-size: 12px;">Physical (Actual)</span>
                    <span style="display: inline-block; width: 20px; height: 3px; background: #e74c3c; margin-left: 15px; margin-right: 5px; border-style: dashed;"></span>
                    <span style="font-size: 12px;">Digital (Predicted)</span>
                </div>
                <div id="prediction-status" style="font-size: 12px; padding: 5px 10px; border-radius: 4px; background: var(--accent); color: white;">
                    Loading...
                </div>
            </div>
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
        
        // ===== PREDICTION CHART (IMPROVED) =====
        const ctx = document.getElementById('predictionChart');
        let predictionChart = null;
        const historyPoints = 60;    // 60 seconds of history
        const predictionPoints = 60; // 60 seconds of prediction
        const totalPoints = historyPoints + predictionPoints;
        
        // Data storage
        let actualHistory = [];  // Past 60 seconds
        let lastPrediction = null;
        
        function initChart() {
            // Create labels: -60s to +60s
            const labels = [];
            for (let i = -historyPoints; i <= predictionPoints; i++) {
                if (i === 0) labels.push('NOW');
                else if (i % 15 === 0) labels.push(i + 's');
                else labels.push('');
            }
            
            predictionChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Physical (Actual)',
                            data: new Array(totalPoints + 1).fill(null),
                            borderColor: '#007bff',
                            backgroundColor: 'rgba(0, 123, 255, 0.15)',
                            fill: true,
                            tension: 0.3,
                            pointRadius: 0,
                            borderWidth: 2.5
                        },
                        {
                            label: 'Digital (Predicted)',
                            data: new Array(totalPoints + 1).fill(null),
                            borderColor: '#e74c3c',
                            borderDash: [8, 4],
                            backgroundColor: 'rgba(231, 76, 60, 0.15)',
                            fill: true,
                            tension: 0.3,
                            pointRadius: 0,
                            borderWidth: 2.5
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: { intersect: false, mode: 'index' },
                    scales: {
                        x: {
                            display: true,
                            title: { display: true, text: 'Time (seconds)', font: { size: 11 } },
                            grid: { 
                                color: (ctx) => ctx.tick.label === 'NOW' ? 'rgba(255,255,255,0.5)' : 'rgba(128,128,128,0.2)',
                                lineWidth: (ctx) => ctx.tick.label === 'NOW' ? 2 : 1
                            }
                        },
                        y: {
                            display: true,
                            title: { display: true, text: 'Bandwidth (KB/s)', font: { size: 11 } },
                            beginAtZero: true,
                            grid: { color: 'rgba(128,128,128,0.2)' }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => ctx.dataset.label + ': ' + (ctx.parsed.y ? ctx.parsed.y.toFixed(1) : '0') + ' KB/s'
                            }
                        }
                    }
                }
            });
        }
        
        async function updateChart() {
            try {
                const response = await fetch('/api/prediction');
                const data = await response.json();
                
                if (data.error && !data.current_traffic) {
                    document.getElementById('prediction-status').textContent = 'Waiting...';
                    document.getElementById('prediction-status').style.background = '#666';
                    return;
                }
                
                // Update status badge
                const statusEl = document.getElementById('prediction-status');
                const conf = data.confidence ? (data.confidence * 100).toFixed(0) : '50';
                statusEl.textContent = data.state + ' (' + conf + '%) → ' + data.prediction_horizon + 's ahead';
                statusEl.style.background = data.color || '#666';
                
                // Convert to KB/s
                const actualKB = data.current_traffic ? data.current_traffic / 1000 : 0;
                const predictedKB = data.estimated_bandwidth ? data.estimated_bandwidth / 1000 : actualKB;
                
                // Add to history
                actualHistory.push(actualKB);
                if (actualHistory.length > historyPoints) {
                    actualHistory.shift();
                }
                
                // Build actual data array: history up to NOW, then null for future
                const actualData = [];
                for (let i = 0; i < historyPoints - actualHistory.length; i++) {
                    actualData.push(null);
                }
                actualData.push(...actualHistory);
                actualData.push(actualHistory[actualHistory.length - 1]); // NOW point
                for (let i = 0; i < predictionPoints; i++) {
                    actualData.push(null); // No actual data for future
                }
                
                // Build prediction data array with TREND (not flat!)
                // Calculate recent trend from last 10 actual points
                const predData = [];
                for (let i = 0; i < historyPoints; i++) {
                    predData.push(null);
                }
                
                const currentVal = actualHistory[actualHistory.length - 1] || 0;
                const targetVal = predictedKB;
                
                // Calculate trend (slope) from recent history
                let trend = 0;
                if (actualHistory.length >= 5) {
                    const recent = actualHistory.slice(-10);
                    const start = recent.slice(0, 3).reduce((a,b) => a+b, 0) / 3;
                    const end = recent.slice(-3).reduce((a,b) => a+b, 0) / 3;
                    trend = (end - start) / recent.length;
                }
                
                // Project forward with trend, easing toward predicted state
                predData.push(currentVal); // NOW point
                for (let i = 1; i <= predictionPoints; i++) {
                    // Blend between trend continuation and target state
                    const trendValue = currentVal + trend * i;
                    const blendFactor = Math.min(1, i / 30); // Ease over 30 seconds
                    const blendedValue = trendValue * (1 - blendFactor) + targetVal * blendFactor;
                    
                    // Add some randomness for natural look
                    const noise = (Math.random() - 0.5) * targetVal * 0.1;
                    predData.push(Math.max(0, blendedValue + noise));
                }
                
                // Update chart data
                predictionChart.data.datasets[0].data = actualData;
                predictionChart.data.datasets[1].data = predData;
                predictionChart.data.datasets[1].borderColor = data.color || '#e74c3c';
                predictionChart.data.datasets[1].backgroundColor = (data.color || '#e74c3c') + '25';
                
                predictionChart.update('none');
                
            } catch (err) {
                console.error('Chart update error:', err);
            }
        }
        
        // Initialize chart
        if (ctx) {
            initChart();
            updateChart();
            // Update chart every 2 seconds (less laggy)
            setInterval(updateChart, 2000);
        }
        
        // No page refresh - chart handles updates via AJAX
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
        db_stats_data = {'traffic_stats': 0, 'flow_stats': 0, 'hosts': 0, 'predictions': 0}
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


@app.route('/api/prediction')
def api_prediction():
    """
    API endpoint for real-time traffic state prediction.
    Returns: state, confidence, estimated_bandwidth, color
    """
    global state_predictor
    
    try:
        db = get_db()
        
        # Get recent traffic data from database
        traffic_history = db.get_traffic_history(minutes=2)
        
        if not traffic_history or len(traffic_history) < 2:
            return jsonify({
                'error': 'Insufficient traffic data',
                'state': 'UNKNOWN',
                'confidence': 0,
                'estimated_bandwidth': 0,
                'current_traffic': 0,
                'color': '#666666'
            })
        
        # CRITICAL FIX v2: Calculate deltas PER PORT, then sum
        # Each (dpid, port) has its own cumulative counter
        # We need to: 1) Group by (dpid,port,timestamp), 2) Calc delta per port, 3) Sum per timestamp
        
        from collections import defaultdict
        
        # Structure: {(dpid, port): {timestamp: bytes}}
        port_data = defaultdict(lambda: defaultdict(int))
        
        for entry in traffic_history:
            dpid = entry.get('dpid', 0)
            port = entry.get('port_no', 0)
            ts = entry.get('timestamp', '')
            total_bytes = entry.get('tx_bytes', 0) + entry.get('rx_bytes', 0)
            port_data[(dpid, port)][ts] = total_bytes
        
        # Get all unique timestamps
        all_timestamps = set()
        for port_key in port_data:
            all_timestamps.update(port_data[port_key].keys())
        sorted_timestamps = sorted(all_timestamps)
        
        # Calculate delta per timestamp (sum of deltas across all ports)
        bytes_per_second = []
        
        for i in range(1, len(sorted_timestamps)):
            prev_ts = sorted_timestamps[i-1]
            curr_ts = sorted_timestamps[i]
            
            total_delta = 0
            for port_key in port_data:
                prev_bytes = port_data[port_key].get(prev_ts, 0)
                curr_bytes = port_data[port_key].get(curr_ts, 0)
                
                if prev_bytes > 0 and curr_bytes >= prev_bytes:
                    delta = curr_bytes - prev_bytes
                    total_delta += delta
            
            bytes_per_second.append(total_delta)
        
        if not bytes_per_second:
            bytes_per_second = [0]
        
        current_traffic = bytes_per_second[-1] if bytes_per_second else 0
        avg_traffic = np.mean(bytes_per_second) if bytes_per_second else 0
        
        # Debug logging
        print(f"[API] Timestamps: {len(sorted_timestamps)}, Deltas: {len(bytes_per_second)}, "
              f"Current: {current_traffic:,.0f} B/s, Avg: {avg_traffic:,.0f} B/s")
        
        # If we have a predictor with enough data, use it
        if state_predictor and len(bytes_per_second) >= 10:
            # Pad if needed
            seq_len = state_predictor.sequence_length
            if len(bytes_per_second) < seq_len:
                padding = [bytes_per_second[0]] * (seq_len - len(bytes_per_second))
                input_data = padding + bytes_per_second
                print(f"[ML] Padded: {len(bytes_per_second)} -> {len(input_data)} samples")
            else:
                input_data = bytes_per_second[-seq_len:]
            
            historical_data = np.array(input_data, dtype=np.float32)
            
            # Debug: show input stats
            print(f"[ML] Input stats: min={historical_data.min():.0f}, max={historical_data.max():.0f}, "
                  f"mean={historical_data.mean():.0f}, std={historical_data.std():.0f}")
            
            result = state_predictor.predict(historical_data)
            
            # Debug: show prediction
            print(f"[ML] Prediction: {result['state']} ({result['confidence']*100:.1f}%), "
                  f"Est BW: {result['estimated_bandwidth']:,.0f} B/s")
            
            result['current_traffic'] = current_traffic
            result['avg_traffic'] = avg_traffic  # Add avg for display
            result['timestamp'] = time.time()
            result['model_used'] = True
            return jsonify(result)
        else:
            reason = "no predictor" if not state_predictor else f"only {len(bytes_per_second)} samples"
            print(f"[FALLBACK] Using threshold-based: {reason}")
            # Fallback: simple threshold-based estimation
            # These thresholds are for bytes/second
            if avg_traffic < 100000:  # < 100 KB/s
                state, color = 'NORMAL', '#27ae60'
            elif avg_traffic < 500000:  # < 500 KB/s
                state, color = 'ELEVATED', '#f39c12'
            elif avg_traffic < 1000000:  # < 1 MB/s
                state, color = 'HIGH', '#e67e22'
            else:  # > 1 MB/s
                state, color = 'CRITICAL', '#e74c3c'
            
            return jsonify({
                'state': state,
                'state_id': ['NORMAL', 'ELEVATED', 'HIGH', 'CRITICAL'].index(state),
                'confidence': 0.5,
                'estimated_bandwidth': avg_traffic,
                'current_traffic': current_traffic,
                'color': color,
                'prediction_horizon': 60,
                'timestamp': time.time(),
                'model_loaded': state_predictor is not None
            })
            
    except Exception as e:
        print(f"Prediction API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'state': 'ERROR',
            'confidence': 0,
            'estimated_bandwidth': 0,
            'color': '#666666'
        })


def start_web_server(host='0.0.0.0', port=5000):
    app.run(host=host, port=port)
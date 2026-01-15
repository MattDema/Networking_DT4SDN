// --- VIS.JS TOPOLOGY VISUALIZATION ---
// Initialize these at the top level so they are available to safe scopes
let network = null;
let nodesDataSet = new vis.DataSet();
let edgesDataSet = new vis.DataSet();

// --- THEME HANDLING ---
function updateThemeUI(theme) {
    if (!document.body) return; 

    const isLight = theme === 'light';
    document.body.setAttribute('data-theme', theme);
    const icon = document.getElementById('theme-icon');
    const text = document.getElementById('theme-text');
    if(icon && text) {
        icon.textContent = isLight ? '🌙' : '☀️';
        text.textContent = isLight ? 'Dark' : 'Light';
    }
    
    // Update graph colors if network exists
    if (network) {
        const textColor = isLight ? '#333333' : '#eeeeee';
        const edgeColor = isLight ? '#95a5a6' : '#666666';
        
        network.setOptions({
            nodes: { font: { color: textColor } },
            edges: { color: edgeColor }
        });
    }
}

function toggleTheme() {
    const currentTheme = document.body.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    localStorage.setItem('theme', newTheme);
    updateThemeUI(newTheme);
}

// Apply theme safely when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    updateThemeUI(savedTheme);
});

function drawTopology(rawTopo) {
    if (!rawTopo) {
        console.error("DEBUG: rawTopo is null or undefined");
        return;
    }
    console.log("DEBUG: Drawing topology...", rawTopo);

    nodesDataSet.clear();
    edgesDataSet.clear();
    
    // Check if rawTopo is valid
    if (!rawTopo || (!rawTopo.switches && !rawTopo.hosts)) {
        console.warn("DEBUG: No topology data to draw.");
        return;
    }
    
    // 1. Add Switches
    if (rawTopo.switches && Array.isArray(rawTopo.switches)) {
        rawTopo.switches.forEach(sw => {
            nodesDataSet.add({
                id: sw, 
                label: sw ? sw.toUpperCase() : "Unknown", 
                group: 'switch',
                image: 'https://img.icons8.com/ios-filled/50/000000/switch.png',
                shape: 'image'
            });
        });
    }

    // 2. Add Hosts
    if (rawTopo.hosts && Array.isArray(rawTopo.hosts)) {
        rawTopo.hosts.forEach(host => {
            nodesDataSet.add({ 
                id: host, 
                label: host ? host.toUpperCase() : "Host", 
                group: 'host', 
                shape: 'dot', 
                color: '#e74c3c' 
            });
        });
    }

    // 3. Add Links
    if (rawTopo.links && Array.isArray(rawTopo.links)) {
        rawTopo.links.forEach(link => {
            if (link.length >= 2) {
                let fromNode = link[0];
                let toNode = link[1];
                let edgeId = fromNode + "-" + toNode; 
                edgesDataSet.add({ id: edgeId, from: fromNode, to: toNode });
            }
        });
    }
}

function initNetwork(rawTopoParams) {
    console.log("DEBUG: Initializing Network with params:", rawTopoParams);
    
    // Try to get container
    const container = document.getElementById('mynetwork');
    if (!container) {
        console.error("DEBUG: Container #mynetwork not found via ID!");
        return;
    }

    // Initialize dataset
    drawTopology(rawTopoParams);

    const currentTheme = localStorage.getItem('theme') || 'light';
    const isLight = currentTheme === 'light';
    const textColor = isLight ? '#333333' : '#eeeeee';
    const edgeColor = isLight ? '#95a5a6' : '#666666';

    const data = { nodes: nodesDataSet, edges: edgesDataSet };
    const options = {
        layout: { 
            hierarchical: false, 
            improvedLayout: true 
        },
        physics: {
            enabled: true,
            stabilization: {
                enabled: true,
                iterations: 500
            },
            barnesHut: { 
                gravitationalConstant: -3000, 
                springConstant: 0.04, 
                springLength: 95 
            }
        },
        edges: { 
            color: edgeColor, 
            width: 2,
            smooth: { type: 'continuous' }
        },
        nodes: {
            font: { size: 14, color: textColor }
        }
    };
    
    try {
        network = new vis.Network(container, data, options);
        console.log("DEBUG: Vis.js Network created successfully.");
    } catch (e) {
        console.error("DEBUG: Failed to create Vis.js network:", e);
    }
}


// Auto Refresh logic
setInterval(() => { 
    // Only reload if user isn't interacting (optional, but for now simple reload)
    window.location.reload(); 
}, 10000);

// Flow Table population (example function)
// This function is just an example and may not directly work with your data structures
function populateFlowTable(dpid, flowList) {
    let html = '';
    const tableBody = document.getElementById('flowTableBody');
    if (!tableBody) return;

    flowList.forEach(flow => {
        html += `
            <tr>
                <td>S${dpid}</td>
                <td><code>${flow.match}</code></td>
                <td><strong>${flow.packet_count}</strong></td>
            </tr>
        `;
    });
    
    // If no flows, show a message
    if (!hasFlows) {
        html = '<tr><td colspan="3" style="text-align:center; padding: 20px;">No active flows</td></tr>';
    }

    tableBody.innerHTML = html;
}

// ===== MODEL SWITCHING =====
async function switchModel(scenario) {
    const statusEl = document.getElementById('model-status');
    statusEl.textContent = '⏳';
    
    try {
        const response = await fetch('/api/models/switch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({scenario: scenario})
        });
        const data = await response.json();
        
        if (data.classifier_switched && data.seq2seq_switched) {
            statusEl.textContent = '✓';
            statusEl.style.color = '#28a745';
        } else {
            statusEl.textContent = '⚠️';
            statusEl.style.color = 'orange';
        }
    } catch (err) {
        statusEl.textContent = '❌';
        statusEl.style.color = '#dc3545';
        console.error('Switch error:', err);
    }
    
    setTimeout(() => { statusEl.textContent = ''; }, 3000);
}

async function loadCurrentModel() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        if (data.classifier && data.classifier.current_model) {
            const select = document.getElementById('scenario-select');
            if(select) select.value = data.classifier.current_model;
        }
    } catch (err) {
        console.error('Could not load current model:', err);
    }
}

// ===== PREDICTION CHART =====
let predictionChart = null;
const historyPoints = 60;
const predictionPoints = 60;
let actualHistory = [];

function initChart() {
    const ctx = document.getElementById('predictionChart');
    if (!ctx) return;

    const labels = Array(historyPoints + predictionPoints + 1).fill('');
    labels[historyPoints] = 'NOW';

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Physical (Actual)',
                    data: [],
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 2
                },
                {
                    label: 'Digital (Predicted)',
                    data: [],
                    borderColor: '#e74c3c',
                    borderDash: [5, 5],
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { intersect: false, mode: 'index' },
            scales: {
                x: { grid: { display: false } }, // Simplify X axis
                y: { beginAtZero: true, title: { display: true, text: 'KB/s' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

async function updateChart() {
    if (!predictionChart) return;

    try {
        const response = await fetch('/api/prediction');
        const data = await response.json();
        
        if (data.error && !data.current_traffic) return;
        
        // Update UI Badge
        const statusEl = document.getElementById('prediction-status');
        if(statusEl) {
            const conf = data.confidence ? (data.confidence * 100).toFixed(0) : '50';
            statusEl.textContent = `${data.state} (${conf}%) → ${data.prediction_horizon}s`;
            statusEl.style.background = data.color || '#666';
        }

        // Processing Logic
        const actualKB = data.current_traffic ? data.current_traffic / 1000 : 0;
        
        // Manage History Array
        actualHistory.push(actualKB);
        if (actualHistory.length > historyPoints) actualHistory.shift();
        
        // Prepare Dataset 1: Actual
        const actualData = new Array(historyPoints + predictionPoints + 1).fill(null);
        // Fill history (from index 0 to historyPoints)
        actualHistory.forEach((val, idx) => {
            // Calculate offset so the last item is at index 'historyPoints' (NOW)
            const offset = historyPoints - actualHistory.length + 1 + idx;
             if(offset >= 0) actualData[offset] = val;
        });

        // Prepare Dataset 2: Prediction
        const predData = new Array(historyPoints + predictionPoints + 1).fill(null);
        // Start prediction line from "NOW"
        const currentVal = actualHistory[actualHistory.length - 1] || 0;
        predData[historyPoints] = currentVal;

        if (data.future_values && data.future_values.length > 0) {
            data.future_values.forEach((val, idx) => {
                if (idx < predictionPoints) {
                    predData[historyPoints + 1 + idx] = val / 1000;
                }
            });
        }

        predictionChart.data.datasets[0].data = actualData;
        predictionChart.data.datasets[1].data = predData;
        predictionChart.update('none');

    } catch (err) {
        console.error('Chart update error:', err);
    }
}

// Init everything when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    loadCurrentModel();
    initChart();
    setInterval(updateChart, 2000);
});
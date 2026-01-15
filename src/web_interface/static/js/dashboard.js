let network = null;
let nodesDataSet = new vis.DataSet();
let edgesDataSet = new vis.DataSet();
let selectedLink = null; 

// --- 1. THEME & INIT ---
function updateThemeUI(theme) {
    if (!document.body) return; 
    const isLight = theme === 'light';
    document.body.setAttribute('data-theme', theme);
    if (network) {
        network.setOptions({
            nodes: { font: { color: isLight ? '#333' : '#eee' } },
            edges: { color: isLight ? '#95a5a6' : '#666' }
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    updateThemeUI(savedTheme);
    
    // Create UI
    createLinkSelectorUI();
    initChart();
    initNetwork(); 
    
    // Start Data Loop
    refreshDashboard(); 
    setInterval(updateChart, 2000);       
    setInterval(refreshDashboard, 3000);  
});

// --- 2. LINK SELECTOR UI ---
function createLinkSelectorUI() {
    let container = document.getElementById('controls-area'); 
    
    // Fallback if you didn't add the div in Step 1
    if (!container) {
        container = document.querySelector('.card-header'); 
    }

    if (container && !document.getElementById('link-select')) {
        const wrapper = document.createElement('div');
        wrapper.className = "d-flex align-items-center";
        wrapper.innerHTML = `
            <label class="me-2 fw-bold small mb-0">Monitor Link:</label>
            <select id="link-select" class="form-select form-select-sm" style="width: auto; min-width: 220px;">
                <option value="">Global Network Load (Aggregate)</option>
            </select>
        `;
        container.appendChild(wrapper);

        document.getElementById('link-select').addEventListener('change', (e) => {
            selectedLink = e.target.value;
            // Clear chart on switch
            actualHistory = []; 
            if(predictionChart) {
                predictionChart.data.datasets[0].data = [];
                predictionChart.data.datasets[1].data = [];
                predictionChart.update();
            }
        });
    }
}

// --- 3. DASHBOARD REFRESH ---
async function refreshDashboard() {
    try {
        const response = await fetch(`/api/dashboard-data?t=${new Date().getTime()}`);
        if (!response.ok) return;
        
        const data = await response.json();
        if (data.error) return;

        // Topology
        if (data.topology && data.hosts) updateVisNetwork(data.topology, data.hosts);
        
        // Flows & Link Options
        if (data.flows) {
            updateFlowTable(data.flows);
            populateLinkSelector(data.flows);
        }

    } catch (err) { console.error("Refresh failed:", err); }
}

// Populate dropdown only once
let linksPopulated = false;
function populateLinkSelector(flows) {
    if (linksPopulated) return;
    
    const select = document.getElementById('link-select');
    if (!select) return;

    // Standard links for your topology
    const candidates = [
        's1-eth1', 's1-eth2', 's1-eth3',
        's2-eth1', 's2-eth2', 's2-eth3',
        's3-eth1', 's3-eth2', 's3-eth3'
    ];

    candidates.forEach(link => {
        if (!select.querySelector(`option[value="${link}"]`)) {
            const opt = document.createElement('option');
            opt.value = link;
            opt.text = link.toUpperCase();
            select.appendChild(opt);
        }
    });
    linksPopulated = true;
}

// --- 4. VISUALIZATION UPDATES ---
function updateVisNetwork(topo, hosts) {
    if (!nodesDataSet || !edgesDataSet) return;
    const newNodes = [];
    const newEdges = [];

    if (topo.switches) {
        topo.switches.forEach(sw => newNodes.push({
            id: sw, label: sw.toUpperCase(), group: 'switch',
            image: 'https://img.icons8.com/ios-filled/50/000000/switch.png', shape: 'image'
        }));
    }
    if (hosts) {
        hosts.forEach(h => {
            newNodes.push({ id: h.mac, label: 'Host', group: 'host', shape: 'dot', color: '#e74c3c' });
            newEdges.push({ id: `link_${h.mac}_s${h.dpid}`, from: h.mac, to: `s${h.dpid}` });
        });
    }
    if (topo.links) {
        topo.links.forEach(l => {
            if(l.length>=2) newEdges.push({ id: `link_${l[0]}_${l[1]}`, from: l[0], to: l[1] });
        });
    }
    nodesDataSet.update(newNodes);
    edgesDataSet.update(newEdges);
}

function updateFlowTable(allFlows) {
    const tableBody = document.getElementById('flowTableBody');
    if (!tableBody) return;
    let html = '';
    
    Object.keys(allFlows).forEach(dpid => {
        const flows = allFlows[dpid];
        flows.sort((a, b) => b.packet_count - a.packet_count);
        flows.forEach(flow => {
            if (flow.priority === 0 && flow.packet_count === 0) return;
            let m = JSON.stringify(flow.match||{}).replace(/[{"}]/g,'').replace(/,/g,', ');
            if(m.length>40) m=m.substring(0,40)+'...';
            html += `<tr><td>S${dpid}</td><td class="font-monospace"><small>${m}</small></td><td><strong>${flow.packet_count}</strong></td></tr>`;
        });
    });
    if (!html) html = '<tr><td colspan="3" class="text-center text-muted">No active flows</td></tr>';
    tableBody.innerHTML = html;
}

function initNetwork() {
    const container = document.getElementById('mynetwork');
    if (!container) return;
    network = new vis.Network(container, 
        { nodes: nodesDataSet, edges: edgesDataSet }, 
        { 
            layout: { improvedLayout: true },
            physics: { enabled: true, stabilization: { iterations: 100 } },
            edges: { width: 2 },
            nodes: { font: { size: 14 } }
        }
    );
}

// --- 5. CHART LOGIC ---
let predictionChart = null;
const points = 60;
let actualHistory = [];

function initChart() {
    const ctx = document.getElementById('predictionChart');
    if (!ctx) return;
    const labels = Array(points * 2 + 1).fill('');
    labels[points] = 'NOW';

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                { label: 'Actual', data: [], borderColor: '#007bff', backgroundColor: 'rgba(0,123,255,0.1)', fill: true, tension: 0.3, pointRadius: 0 },
                { label: 'Predicted', data: [], borderColor: '#e74c3c', borderDash: [5,5], backgroundColor: 'rgba(231,76,60,0.1)', fill: true, tension: 0.3, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { intersect: false, mode: 'index' },
            scales: { x: { display: false }, y: { beginAtZero: true, title: {display: true, text: 'KB/s'} } },
            plugins: { legend: { display: false } }
        }
    });
}

async function updateChart() {
    if (!predictionChart) return;
    try {
        let url = '/api/prediction';
        if (selectedLink) url += `?link_id=${selectedLink}`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.error && !data.current_traffic) return;
        
        // Status Badge
        const statusEl = document.getElementById('prediction-status');
        if(statusEl) {
            const conf = data.confidence ? (data.confidence*100).toFixed(0) : '0';
            const lbl = selectedLink ? selectedLink.toUpperCase() : "AGGREGATE";
            statusEl.textContent = `${lbl}: ${data.state} (${conf}%)`;
            statusEl.style.background = data.color || '#666';
        }

        // Data Update
        const val = data.current_traffic ? data.current_traffic/1000 : 0;
        actualHistory.push(val);
        if (actualHistory.length > points) actualHistory.shift();

        const actData = new Array(points * 2 + 1).fill(null);
        actualHistory.forEach((v, i) => actData[points - actualHistory.length + 1 + i] = v);

        const predData = new Array(points * 2 + 1).fill(null);
        predData[points] = actualHistory[actualHistory.length-1]||0;
        
        if (data.future_values) {
            data.future_values.forEach((v, i) => {
                if (i < points) predData[points + 1 + i] = v/1000;
            });
        }

        predictionChart.data.datasets[0].data = actData;
        predictionChart.data.datasets[1].data = predData;
        predictionChart.update('none');
    } catch (e) { console.error(e); }
}


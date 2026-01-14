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
                <td><code>${flow.instructions}</code></td>
                <td><strong>${flow.packet_count}</strong></td>
            </tr>
        `;
    });

    tableBody.innerHTML = html;
}
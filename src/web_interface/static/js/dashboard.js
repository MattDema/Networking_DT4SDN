// Load saved theme preference
const savedTheme = localStorage.getItem('theme') || 'light';
document.body.setAttribute('data-theme', savedTheme);
updateThemeButton(savedTheme);

function toggleTheme() {
    const body = document.body;
    const currentTheme = body.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    body.setAttribute('data-theme', newTheme);
    document.getElementById('theme-icon').textContent = newTheme === 'light' ? '🌙' : '☀️';
    document.getElementById('theme-text').textContent = newTheme === 'light' ? 'Dark' : 'Light';
}

// --- VIS.JS TOPOLOGY VISUALIZATION ---
let network = null;
let nodesDataSet = new vis.DataSet();
let edgesDataSet = new vis.DataSet();

function drawTopology(rawTopo) {
    if (!rawTopo) {
        console.error("DEBUG: rawTopo is null or undefined");
        return;
    }
    console.log("DEBUG: Drawing topology with:", rawTopo);

    nodesDataSet.clear();
    edgesDataSet.clear();
    
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
                shape: 'dot', user: 'host',
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
    console.log("DEBUG: Initializing Network...");
    
    const container = document.getElementById('mynetwork');
    if (!container) {
        console.error("DEBUG: Container #mynetwork not found!");
        return;
    }

    // Initialize dataset FIRST
    drawTopology(rawTopoParams);

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
                iterations: 1000,
                updateInterval: 25
            },
            barnesHut: { 
                gravitationalConstant: -3000, 
                springConstant: 0.04, 
                springLength: 95 
            }
        },
        edges: { 
            color: "#95a5a6", 
            width: 2,
            smooth: {
                type: 'continuous'
            }
        },
        nodes: {
            font: { size: 14, color: '#333' },
            borderWidth: 2
        }
    };
    
    // Create network
    network = new vis.Network(container, data, options);
    
    // Force fit to screen
    network.once("stabilizationIterationsDone", function() {
        console.log("DEBUG: Stabilization done, fitting.");
        network.fit();
    });
}

// Auto Refresh
setInterval(() => { window.location.reload(); }, 10000);
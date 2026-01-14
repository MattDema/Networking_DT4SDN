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
    nodesDataSet.clear();
    edgesDataSet.clear();
    
    // 1. Add Switches
    if (rawTopo.switches) {
        rawTopo.switches.forEach(sw => {
            nodesDataSet.add({
                id: sw, 
                label: sw.toUpperCase(), 
                group: 'switch',
                image: 'https://img.icons8.com/ios-filled/50/000000/switch.png',
                shape: 'image'
            });
        });
    }

    // 2. Add Hosts
    if (rawTopo.hosts) {
        rawTopo.hosts.forEach(host => {
            nodesDataSet.add({ id: host, label: host.toUpperCase(), group: 'host', shape: 'dot', color: '#e74c3c' });
        });
    }

    // 3. Add Links
    if (rawTopo.links) {
        rawTopo.links.forEach(link => {
            let fromNode = link[0];
            let toNode = link[1];
            let edgeId = fromNode + "-" + toNode; 
            edgesDataSet.add({ id: edgeId, from: fromNode, to: toNode });
        });
    }
}

function initNetwork(rawTopoParams) {
    drawTopology(rawTopoParams);

    const container = document.getElementById('mynetwork');
    const data = { nodes: nodesDataSet, edges: edgesDataSet };
    const options = {
        layout: { hierarchical: false, improvedLayout: true },
        physics: {
            stabilization: true,
            barnesHut: { gravitationalConstant: -2000, springConstant: 0.04, springLength: 95 }
        },
        edges: { color: "#95a5a6", width: 2 }
    };
    
    if (container) {
        network = new vis.Network(container, data, options);
    }
}

// Auto Refresh
setInterval(() => { window.location.reload(); }, 10000);
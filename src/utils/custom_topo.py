from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.link import TCLink
import json
import os
import requests
import time

class CustomTopo(Topo):
    def __init__(self):
        Topo.__init__(self)

        host_config = dict(inNamespace=True)
        link_config = dict()
        # Create 3 switches
        sconfig = {"dpid": "%016x" % 1}
        self.addSwitch("s1", **sconfig)
        sconfig = {"dpid": "%016x" % 2}
        self.addSwitch("s2", **sconfig)
        sconfig = {"dpid": "%016x" % 3}
        self.addSwitch("s3", **sconfig)
        
        # Create 3 hosts
        self.addHost("h1", **host_config)
        self.addHost("h2", **host_config)
        self.addHost("h3", **host_config)

        # Link switches linearly
        self.addLink("s1", "s2", **link_config)
        self.addLink("s2", "s3", **link_config)

        # Connect hosts to switches
        self.addLink("h1", "s1", **link_config)
        self.addLink("h2", "s2", **link_config)
        self.addLink("h3", "s3", **link_config)

topos = {"customtopo": (lambda: CustomTopo())}

def save_topology_metadata(topo, topo_type="linear_manual"): # Default type
    data = {
        "type": topo_type,
        "switches": topo.switches(),
        "hosts": topo.hosts(),
        "links": topo.links(sort=True)
    }
    
    # 1. Save locally (for backup/debugging)
    with open("current_topology.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"Topology metadata saved to {os.path.abspath('current_topology.json')}")

    # 2. Push to Ryu Controller
    # Wait a moment to ensure Controller is ready
    time.sleep(1) 
    try:
        url = "http://127.0.0.1:8080/topology/metadata"
        requests.post(url, json=data, timeout=2)
        print(f"✓ Topology metadata sent to Controller ({url})")
    except Exception as e:
        print(f"⚠ Warning: Could not push topology to Ryu: {e}")
        print("  (Is the controller running? sudo ryu-manager ...)")


def main():
    topo = CustomTopo()

    # Save and Push Topology Info
    save_topology_metadata(topo, "linear_manual")

    net = Mininet(
        topo=topo,
        switch=OVSKernelSwitch,
        build=False,
        autoSetMacs=True,
        autoStaticArp=True,
        link=TCLink,
    )
    controller = RemoteController("c1", ip="127.0.0.1", port=6633)
    net.addController(controller)
    net.build()
    net.start()
    CLI(net)
    net.stop()

if __name__ == "__main__":
    main()
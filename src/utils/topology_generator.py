# src/utils/topology_generator.py
"""
Generate custom Mininet topologies from user specifications
"""

INCLUDES = """from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.link import TCLink
import json
import os
import requests
import time
"""

TOPO_CLASS_START = """
class CustomTopo(Topo):
    def __init__(self):
        Topo.__init__(self)
        
        host_config = dict(inNamespace=True)
        link_config = dict()
"""

MAIN_FUNCTION = """
topos = {"customtopo": (lambda: CustomTopo())}

def save_topology_metadata(topo, topo_type):
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

    # 2. Push to Ryu Controller (so Dashboard can see it)
    # We wait a brief moment to ensure Ryu is ready if started simultaneously
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
    
    # Save topology info for Dashboard
    save_topology_metadata(topo, "{TOPOLOGY_TYPE}")

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
"""


def generate_linear_topology(num_switches: int, num_hosts: int) -> str:
    """
    Linear topology: s1--s2--s3--s4
    Hosts distributed evenly across switches
    """
    code = ""
    
    # Create switches
    code += f"        # Create {num_switches} switches\n"
    for i in range(num_switches):
        code += f'        sconfig = {{"dpid": "%016x" % {i+1}}}\n'
        code += f'        self.addSwitch("s{i+1}", **sconfig)\n'
    
    # Create hosts
    code += f"\n        # Create {num_hosts} hosts\n"
    for i in range(num_hosts):
        code += f'        self.addHost("h{i+1}", **host_config)\n'
    
    # Link switches in a line
    code += "\n        # Link switches linearly\n"
    for i in range(num_switches - 1):
        code += f'        self.addLink("s{i+1}", "s{i+2}", **link_config)\n'
    
    # Distribute hosts across switches
    code += "\n        # Connect hosts to switches\n"
    hosts_per_switch = num_hosts // num_switches
    remaining_hosts = num_hosts % num_switches
    
    host_idx = 1
    for switch_idx in range(1, num_switches + 1):
        # Each switch gets at least hosts_per_switch hosts
        hosts_for_this_switch = hosts_per_switch
        if switch_idx <= remaining_hosts:
            hosts_for_this_switch += 1  # Distribute remainder
        
        for _ in range(hosts_for_this_switch):
            if host_idx <= num_hosts:
                code += f'        self.addLink("h{host_idx}", "s{switch_idx}", **link_config)\n'
                host_idx += 1
    
    return code


def generate_star_topology(num_switches: int, num_hosts: int) -> str:
    """
    Star topology: One central switch, others connected to it
    """
    code = ""
    
    # Create switches
    code += f"        # Create {num_switches} switches (s1 is central)\n"
    for i in range(num_switches):
        code += f'        sconfig = {{"dpid": "%016x" % {i+1}}}\n'
        code += f'        self.addSwitch("s{i+1}", **sconfig)\n'
    
    # Create hosts
    code += f"\n        # Create {num_hosts} hosts\n"
    for i in range(num_hosts):
        code += f'        self.addHost("h{i+1}", **host_config)\n'
    
    # Connect all switches to s1 (central switch)
    if num_switches > 1:
        code += "\n        # Connect all switches to central switch (s1)\n"
        for i in range(2, num_switches + 1):
            code += f'        self.addLink("s1", "s{i}", **link_config)\n'
    
    # Distribute hosts
    code += "\n        # Connect hosts to switches\n"
    hosts_per_switch = num_hosts // num_switches
    remaining_hosts = num_hosts % num_switches
    
    host_idx = 1
    for switch_idx in range(1, num_switches + 1):
        hosts_for_this_switch = hosts_per_switch
        if switch_idx <= remaining_hosts:
            hosts_for_this_switch += 1
        
        for _ in range(hosts_for_this_switch):
            if host_idx <= num_hosts:
                code += f'        self.addLink("h{host_idx}", "s{switch_idx}", **link_config)\n'
                host_idx += 1
    
    return code


def generate_tree_topology(num_switches: int, num_hosts: int, depth: int = 2) -> str:
    """
    Tree topology: Hierarchical structure
    """
    code = ""
    
    # Create switches
    code += f"        # Create {num_switches} switches (tree structure)\n"
    for i in range(num_switches):
        code += f'        sconfig = {{"dpid": "%016x" % {i+1}}}\n'
        code += f'        self.addSwitch("s{i+1}", **sconfig)\n'
    
    # Create hosts
    code += f"\n        # Create {num_hosts} hosts\n"
    for i in range(num_hosts):
        code += f'        self.addHost("h{i+1}", **host_config)\n'
    
    # Build tree structure (binary tree)
    code += "\n        # Build tree structure\n"
    for parent in range(1, num_switches + 1):
        left_child = 2 * parent
        right_child = 2 * parent + 1
        
        if left_child <= num_switches:
            code += f'        self.addLink("s{parent}", "s{left_child}", **link_config)\n'
        if right_child <= num_switches:
            code += f'        self.addLink("s{parent}", "s{right_child}", **link_config)\n'
    
    # Connect hosts to leaf switches
    code += "\n        # Connect hosts to leaf switches\n"
    leaf_switches = [s for s in range(1, num_switches + 1) 
                     if 2*s > num_switches]  # Switches with no children
    
    host_idx = 1
    for leaf_switch in leaf_switches:
        if host_idx <= num_hosts:
            code += f'        self.addLink("h{host_idx}", "s{leaf_switch}", **link_config)\n'
            host_idx += 1
    
    # If more hosts than leaf switches, distribute remainder
    while host_idx <= num_hosts:
        for leaf_switch in leaf_switches:
            if host_idx <= num_hosts:
                code += f'        self.addLink("h{host_idx}", "s{leaf_switch}", **link_config)\n'
                host_idx += 1
    
    return code


def generate_mesh_topology(num_switches: int, num_hosts: int) -> str:
    """
    Mesh topology: All switches connected to each other
    """
    code = ""
    
    # Create switches
    code += f"        # Create {num_switches} switches (fully meshed)\n"
    for i in range(num_switches):
        code += f'        sconfig = {{"dpid": "%016x" % {i+1}}}\n'
        code += f'        self.addSwitch("s{i+1}", **sconfig)\n'
    
    # Create hosts
    code += f"\n        # Create {num_hosts} hosts\n"
    for i in range(num_hosts):
        code += f'        self.addHost("h{i+1}", **host_config)\n'
    
    # Create full mesh between switches
    code += "\n        # Create full mesh between switches\n"
    for i in range(1, num_switches + 1):
        for j in range(i + 1, num_switches + 1):
            code += f'        self.addLink("s{i}", "s{j}", **link_config)\n'
    
    # Distribute hosts
    code += "\n        # Connect hosts to switches\n"
    hosts_per_switch = num_hosts // num_switches
    remaining_hosts = num_hosts % num_switches
    
    host_idx = 1
    for switch_idx in range(1, num_switches + 1):
        hosts_for_this_switch = hosts_per_switch
        if switch_idx <= remaining_hosts:
            hosts_for_this_switch += 1
        
        for _ in range(hosts_for_this_switch):
            if host_idx <= num_hosts:
                code += f'        self.addLink("h{host_idx}", "s{switch_idx}", **link_config)\n'
                host_idx += 1
    
    return code


def generate_custom_topology(num_switches: int, num_hosts: int, 
                            topology_type: str = 'linear',
                            output_file: str = 'custom_network.py'):
    """
    Generate Mininet topology script
    """
    
    # Validate inputs
    if num_switches < 1:
        raise ValueError("Need at least 1 switch")
    if num_hosts < 1:
        raise ValueError("Need at least 1 host")
    
    # Generate topology code based on type
    topology_generators = {
        'linear': generate_linear_topology,
        'star': generate_star_topology,
        'tree': generate_tree_topology,
        'mesh': generate_mesh_topology
    }
    
    if topology_type not in topology_generators:
        raise ValueError(f"Unknown topology type: {topology_type}. "
                        f"Choose from: {list(topology_generators.keys())}")
    
    print(f"Generating {topology_type} topology...")
    print(f"  Switches: {num_switches}")
    print(f"  Hosts: {num_hosts}")
    
    # Build complete script
    script = INCLUDES
    script += TOPO_CLASS_START
    script += topology_generators[topology_type](num_switches, num_hosts)
    # REPLACE METADATA PLACEHOLDER
    script += MAIN_FUNCTION.replace("{TOPOLOGY_TYPE}", topology_type)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(script)
    
    print(f"✓ Topology saved to: {output_file}")
    print(f"\nTo run:")
    print(f"  sudo python3 {output_file}")
    
    return output_file


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python topology_generator.py <switches> <hosts> <type> [output_file]")
        print("Types: linear, star, tree, mesh")
        print("\nExample:")
        print("  python topology_generator.py 5 10 linear my_network.py")
        sys.exit(1)
    
    num_switches = int(sys.argv[1])
    num_hosts = int(sys.argv[2])
    topology_type = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else 'custom_network.py'
    
    generate_custom_topology(num_switches, num_hosts, topology_type, output_file)


if __name__ == '__main__':
    main()

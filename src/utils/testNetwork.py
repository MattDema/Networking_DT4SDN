#!/usr/bin/python

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel


class NetworkTopo(Topo):
    def build(self):
        # --- CONFIGURATION ---
        # 10 Mbps limit to match your Orchestrator's Congestion Threshold
        # max_queue_size=100 ensures packets queue up so you can see the queue growing
        link_config = dict(bw=30, delay='5ms', max_queue_size=100)

        # Keep host links fast (optional) or limit them too. 
        # Here we limit everything for simplicity.

        # Creation of switches
        s1 = self.addSwitch('s1', dpid="%016x" % 1)
        s2 = self.addSwitch('s2', dpid="%016x" % 2)
        s3 = self.addSwitch('s3', dpid="%016x" % 3)

        # Creation of hosts
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        h3 = self.addHost('h3')

        # Connect hosts to switches (using the limit)
        self.addLink(h1, s1, **link_config)
        self.addLink(h2, s2, **link_config)
        self.addLink(h3, s3, **link_config)

        # Connect switches (THE BOTTLENECK)
        # Traffic from h1 -> h2 must pass through these 10Mbps links
        self.addLink(s1, s2, **link_config)
        self.addLink(s2, s3, **link_config)


def run():
    topo = NetworkTopo()
    net = Mininet(
        topo=topo,
        switch=OVSKernelSwitch,
        build=False,
        autoSetMacs=True,
        autoStaticArp=True,
        link=TCLink,  # <--- Important: TCLink is required for bw/delay to work
        controller=None  # We add the remote controller manually below
    )

    # Add your Ryu Controller
    c1 = RemoteController("c1", ip="127.0.0.1", port=6633)
    net.addController(c1)

    print("--- Starting Network with 10 Mbps Limits ---")
    net.build()
    net.start()

    # Verify the limits are active
    print("--- Verifying Link Settings ---")
    # This prints the interface config (look for '10Mbit')
    h1 = net.get('h1')
    # h1.cmd('tc class show dev h1-eth0') 

    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run()

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.link import TCLink

class NetworkTopo(Topo):

    def __init__(self):
        Topo.__init__(self)

        #Setup of the network topology

        # Creation of dictionaries for Linux Namespace and host configuration
        host_config = dict(inNamespace=True)
        host_link_config = dict()

        # Creation of switches
        for i in range(3):
            #Hexadecimal notation as OpenFlow Datapath ID is a 64-bit integer
            sconfig = {"dpid": "%016x" % (i + 1)}
            self.addSwitch(f"s{i + 1}", **sconfig)

        # Use them to spawn a host
        self.addHost('h1', **host_config)
        self.addHost('h2', **host_config)


        # Use them to connect the host to a switch
        self.addLink('h1', 's1', **host_link_config)
        self.addLink('h2', 's3', **host_link_config)
        self.addLink('s1', 's2', **host_link_config)
        self.addLink('s2', 's3', **host_link_config)

        # Allows to be launched from command line using sudo mn --custom script.py --topo NetworkTopo
        topos = {"digitaltwintopo": (lambda: NetworkTopo())}

        # Simulation manager
        net = Mininet(
            topo=topo,
            switch=OVSKernelSwitch, #To support OpenFlow
            build=False,
            autoSetMacs=True,
            autoStaticArp=True, #Automatically populate ARP table in order to avoid an ARP request to know hosts' position
            link=TCLink,
        )

        controller = RemoteController("c1", ip="127.0.0.1", port=6633)
        net.addController(controller)
        net.build()
        net.start() #Instatiate the handshake with RYU controller
        CLI(net)
        net.stop()

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.link import TCLink

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

def main():
    topo = CustomTopo()
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
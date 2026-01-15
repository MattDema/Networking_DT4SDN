#!/usr/bin/python

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, Host
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI
import time
import threading
import os
import random
import signal

# --- Configuration ---
OUTPUT_FILE = "network_data_burst.csv"
MONITORED_INTERFACE = "s1-eth2" 
SIMULATION_TIME = 50000  # 30 min
LINK_BW = 30            # Capacity 30 Mbps

class CustomTopo(Topo):
    def build(self):
        host_config = dict(inNamespace=True)
        # Link identical to other scenarios
        link_config = dict(bw=LINK_BW, delay='5ms', loss=0, max_queue_size=100)
        
        self.addSwitch("s1", dpid="%016x" % 1)
        self.addSwitch("s2", dpid="%016x" % 2)
        self.addSwitch("s3", dpid="%016x" % 3)

        self.addHost("h1", **host_config)
        self.addHost("h2", **host_config)
        self.addHost("h3", **host_config)

        self.addLink("s1", "s2", **link_config) 
        self.addLink("s2", "s3", **link_config)
        self.addLink("h1", "s1", **link_config)
        self.addLink("h2", "s2", **link_config)
        self.addLink("h3", "s3", **link_config)

def get_rx_bytes(interface):
    try:
        with open(f'/sys/class/net/{interface}/statistics/rx_bytes', 'r') as f:
            return int(f.read())
    except IOError:
        return 0

def monitor_traffic(stop_event):
    info(f"*** Starting BURST monitoring on {MONITORED_INTERFACE}...\n")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("timestamp,bytes\n")
    
    prev_bytes = get_rx_bytes(MONITORED_INTERFACE)
    start_time = time.time()
    
    while not stop_event.is_set():
        time.sleep(1.0) 
        
        current_time = time.time()
        current_bytes = get_rx_bytes(MONITORED_INTERFACE)
        
        delta_bytes = current_bytes - prev_bytes
        elapsed = int(current_time - start_time)
        
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"{elapsed},{delta_bytes}\n")
            
        prev_bytes = current_bytes

def traffic_generator(net):
    h1 = net.get('h1')
    h2 = net.get('h2')
    h3 = net.get('h3') # Vittima

    info("*** Starting Iperf Server on h3...\n")
    h3.cmd('iperf -s -u &') 
    
    info("*** Starting BURST scenario (Quiet -> Spike -> Quiet)...\n")
    
    start_time = time.time()
    
    while (time.time() - start_time) < SIMULATION_TIME:
        elapsed = time.time() - start_time
        
        # --- PHASE 1: CALM (Background Noise) ---
        # Variable interval between bursts (from 10 to 40 seconds)
        # Traffic here is low, under 2 Mbps.
        quiet_duration = random.randint(10, 40)
        
        info(f"[{int(elapsed)}s] Status: Quiet (Next burst in {quiet_duration}s)\n")
        
        # We use a while loop to cover the quiet duration by sending light packets
        end_quiet = time.time() + quiet_duration
        while time.time() < end_quiet:
            # Small heartbeat traffic (0.5M - 1.5M)
            h1.cmd('iperf -c 10.0.0.3 -u -b 1M -t 2')
            time.sleep(2.1) # Wait for iperf to finish

        # --- PHASE 2: BURST (Violent Spike) ---
        # Short duration (1 - 5 seconds)
        # VERY high intensity (sum h1 + h2 >> 10Mbps)
        burst_duration = random.uniform(1.0, 5.0)
        
        info(f"[{int(time.time()-start_time)}s] !!! BURST EVENT !!! ({burst_duration:.1f}s)\n")
        
        # H1 sends 15M (already saturates alone)
        h1.cmd(f'iperf -c 10.0.0.3 -u -b 15M -t {burst_duration} &')
        # H2 sends 15M (finishing blow to the buffer)
        h2.cmd(f'iperf -c 10.0.0.3 -u -b 15M -t {burst_duration} &')
        
        # We wait for the burst to finish + 1 second of "recovery"
        time.sleep(burst_duration + 1)

def stop_all_iperf(net):
    info("*** Cleaning processes...\n")
    for host in net.hosts:
        host.cmd('killall -9 iperf')

def run():
    topo = CustomTopo()
    net = Mininet(topo=topo, link=TCLink, switch=OVSKernelSwitch)
    net.start()

    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(target=monitor_traffic, args=(stop_monitor,))
    monitor_thread.start()

    try:
        traffic_generator(net)
    except KeyboardInterrupt:
        info("*** Manual interruption...\n")
    finally:
        stop_monitor.set()
        monitor_thread.join()
        stop_all_iperf(net)
        net.stop()
        info(f"*** Data saved in {OUTPUT_FILE}\n")

if __name__ == '__main__':
    setLogLevel('info')
    run()

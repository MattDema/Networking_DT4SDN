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

# --- Configurazione ---
OUTPUT_FILE = "network_data_mixed.csv"
MONITORED_INTERFACE = "s1-eth2" 
SIMULATION_TIME = 1800  # 30 minuti
LINK_BW = 10            # 10 Mbps

class CustomTopo(Topo):
    def build(self):
        host_config = dict(inNamespace=True)
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
    info(f"*** Avvio monitoraggio MIXED su {MONITORED_INTERFACE}...\n")
    
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
    h3 = net.get('h3')

    info("*** Avvio Server Iperf su h3...\n")
    h3.cmd('iperf -s -u &') 
    
    start_time = time.time()
    
    info("*** Inizio scenario MIXED (Randomized Events)...\n")
    
    while (time.time() - start_time) < SIMULATION_TIME:
        elapsed = time.time() - start_time
        
        # Scegliamo casualmente cosa succederà adesso
        scenario = random.choices(
            ['NORMAL', 'CONGESTION', 'DDOS', 'BURST'],
            weights=[40, 30, 15, 15], # Probabilità
            k=1
        )[0]
        
        # Durata casuale dell'evento corrente (tra 30s e 90s)
        duration = random.randint(30, 90)
        end_scenario = time.time() + duration
        
        info(f"[{int(elapsed)}s] Switching to: {scenario} (for {duration}s)\n")
        
        while time.time() < end_scenario and (time.time() - start_time) < SIMULATION_TIME:
            
            if scenario == 'NORMAL':
                # Traffico fluido, nessun pacchetto perso
                bw = random.uniform(2, 6)
                h1.cmd(f'iperf -c 10.0.0.3 -u -b {bw:.1f}M -t 5')
                time.sleep(5)
                
            elif scenario == 'CONGESTION':
                # Traffico al limite (8-12 Mbps), code piene
                bw = random.uniform(8, 12)
                h1.cmd(f'iperf -c 10.0.0.3 -u -b {bw:.1f}M -t 2')
                # A volte h2 aggiunge carico extra
                if random.random() > 0.7:
                    h2.cmd('iperf -c 10.0.0.3 -u -b 4M -t 2 &')
                time.sleep(2)
                
            elif scenario == 'DDOS':
                # Saturazione totale e costante
                # Non usiamo loop brevi qui, lanciamo un processo lungo per simulare il plateau
                remaining = int(end_scenario - time.time())
                if remaining > 0:
                    h1.cmd(f'iperf -c 10.0.0.3 -u -b 15M -t {remaining} &')
                    h2.cmd(f'iperf -c 10.0.0.3 -u -b 10M -t {remaining} &')
                    time.sleep(remaining)
                
            elif scenario == 'BURST':
                # Silenzio... CLAMORE... Silenzio
                # 80% del tempo silenzio, 20% spike
                if random.random() > 0.8:
                    info("   -> SPIKE!\n")
                    h1.cmd('iperf -c 10.0.0.3 -u -b 20M -t 2 &')
                    h2.cmd('iperf -c 10.0.0.3 -u -b 20M -t 2 &')
                    time.sleep(3) # Tempo per spike + recovery
                else:
                    # Traffico minimo background
                    h1.cmd('iperf -c 10.0.0.3 -u -b 0.5M -t 2')
                    time.sleep(2)

def stop_all_iperf(net):
    info("*** Pulizia processi...\n")
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
        info("*** Interruzione manuale...\n")
    finally:
        stop_monitor.set()
        monitor_thread.join()
        stop_all_iperf(net)
        net.stop()
        info(f"*** Dati salvati in {OUTPUT_FILE}\n")

if __name__ == '__main__':
    setLogLevel('info')
    run()
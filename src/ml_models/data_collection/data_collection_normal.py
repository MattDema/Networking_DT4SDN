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
OUTPUT_FILE = "network_data_normal.csv" # Nome file diverso per non sovrascrivere
MONITORED_INTERFACE = "s1-eth2" 
SIMULATION_TIME = 1800  # 30 minuti
LINK_BW = 10            # Capacità del link (identica allo scenario congestione)

class CustomTopo(Topo):
    def build(self):
        host_config = dict(inNamespace=True)
        # Il link è lo stesso, ma non lo satureremo
        link_config = dict(bw=LINK_BW, delay='5ms', loss=0, max_queue_size=100)
        
        # Switches
        self.addSwitch("s1", dpid="%016x" % 1)
        self.addSwitch("s2", dpid="%016x" % 2)
        self.addSwitch("s3", dpid="%016x" % 3)

        # Hosts
        self.addHost("h1", **host_config)
        self.addHost("h2", **host_config)
        self.addHost("h3", **host_config)

        # Links
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
    info(f"*** Avvio monitoraggio su {MONITORED_INTERFACE} (Scenario NORMAL)...\n")
    
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
    # h2 non lo usiamo o lo usiamo pochissimo in scenario normale
    h3 = net.get('h3') 

    info("*** Avvio Server Iperf su h3...\n")
    h3.cmd('iperf -s -u &') 
    
    info("*** Inizio traffico NORMALE (Safe Zone < 8Mbps)...\n")
    
    start = time.time()
    
    while (time.time() - start) < SIMULATION_TIME:
        elapsed = time.time() - start
        
        # Logica "Healthy Network": Simuliamo web browsing, streaming video HD, ma niente congestione.
        # Link max = 10M. Noi stiamo sempre sotto.
        mode = random.choices(
            ['idle', 'web', 'streaming', 'file_transfer'], 
            weights=[20, 40, 30, 10], 
            k=1
        )[0]

        target_bw = "0M"

        if mode == 'idle':
            # Traffico di background quasi nullo (keep-alive)
            target_bw = "0.2M"
            
        elif mode == 'web':
            # Navigazione leggera (variabile bassa)
            bw = random.uniform(1, 3)
            target_bw = f"{bw:.1f}M"
            
        elif mode == 'streaming':
            # Streaming video (flusso costante medio)
            bw = random.uniform(3.5, 5.5)
            target_bw = f"{bw:.1f}M"
            
        elif mode == 'file_transfer':
            # Trasferimento file "veloce" ma educato (non satura il link)
            # Rimaniamo tra 6 e 8 Mbps. Il link è 10 Mbps, quindi c'è ancora margine (headroom).
            bw = random.uniform(6.5, 8.0)
            target_bw = f"{bw:.1f}M"

        # Durata casuale dello stato (5-10 secondi)
        duration = random.randint(5, 10)
        
        info(f"[{int(elapsed)}s] Status: OK ({mode}) -> Rate: {target_bw}\n")

        # Inviamo traffico
        h1.cmd(f'iperf -c 10.0.0.3 -u -b {target_bw} -t {duration}')
        
        # Pausa piccolissima casuale tra una richiesta e l'altra (come un utente reale)
        time.sleep(random.uniform(0.1, 1.0))

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
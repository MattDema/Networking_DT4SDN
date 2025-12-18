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
OUTPUT_FILE = "network_data.csv"
MONITORED_INTERFACE = "s1-eth2" # L'interfaccia di s1 che va verso s2 (il collo di bottiglia)
SIMULATION_TIME = 60 # Durata in secondi
LINK_BW = 10 # Bandwidth in Mbps (bassa per facilitare la congestione)

class CustomTopo(Topo):
    def build(self):
        host_config = dict(inNamespace=True)
        # IMPORTANTE: Usiamo TCLink per limitare la banda e creare code
        # bw=10Mbps, delay=5ms, max_queue_size=100 pacchetti
        # Questo assicura che se mandiamo 15Mbps, la coda si riempie -> CONGESTIONE
        link_config = dict(bw=LINK_BW, delay='5ms', loss=0, max_queue_size=100)
        
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
        # s1 <-> s2 (Monitoriamo questo link)
        self.addLink("s1", "s2", **link_config) 
        self.addLink("s2", "s3", **link_config)

        # Connect hosts to switches
        self.addLink("h1", "s1", **link_config)
        self.addLink("h2", "s2", **link_config)
        self.addLink("h3", "s3", **link_config)

def get_rx_bytes(interface):
    """Legge i byte ricevuti dal file system del kernel."""
    try:
        with open(f'/sys/class/net/{interface}/statistics/rx_bytes', 'r') as f:
            return int(f.read())
    except IOError:
        return 0

def monitor_traffic(stop_event):
    """Thread che salva i dati ogni secondo."""
    info(f"*** Avvio monitoraggio su {MONITORED_INTERFACE}...\n")
    
    # Intestazione CSV
    with open(OUTPUT_FILE, "w") as f:
        f.write("timestamp,bytes\n")
    
    prev_bytes = get_rx_bytes(MONITORED_INTERFACE)
    start_time = time.time()
    
    while not stop_event.is_set():
        time.sleep(1.0) # Campionamento ogni secondo
        
        current_time = time.time()
        current_bytes = get_rx_bytes(MONITORED_INTERFACE)
        
        # Calcolo delta (byte trasferiti nell'ultimo secondo)
        delta_bytes = current_bytes - prev_bytes
        elapsed = int(current_time - start_time)
        
        # Scrittura su file
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"{elapsed},{delta_bytes}\n")
            
        prev_bytes = current_bytes
        # Debug print opzionale
        # info(f"Time: {elapsed}s, Bytes: {delta_bytes}\n")

def traffic_generator(net):
    """Genera traffico variabile per saturare la rete."""
    h1 = net.get('h1')
    h2 = net.get('h2') # Useremo h2 come generatore di disturbo
    h3 = net.get('h3') # Destinazione principale

    info("*** Avvio Server Iperf su h3...\n")
    h3.cmd('iperf -s -u &') # Server UDP in background
    
    info("*** Inizio generazione traffico variabile...\n")
    
    start = time.time()
    while time.time() - start < SIMULATION_TIME:
        elapsed = time.time() - start
        
        # Logica per variare il carico (Congestion Pattern)
        # Link capacity è 10Mbps.
        
        # Fase 1: Traffico normale (5 Mbps)
        target_bw = "5M"
        
        # Fase 2: Saturazione leggera (10 Mbps) - dopo 10 secondi
        if 10 < elapsed <= 20:
            target_bw = "10M"
            
        # Fase 3: Congestione Pesante (15 Mbps) - Sovraccarico del 50%
        elif 20 < elapsed <= 30:
            target_bw = "15M"
            
        # Fase 4: Pausa/Calo (2 Mbps)
        elif 30 < elapsed <= 40:
            target_bw = "2M"
            
        # Fase 5: Burst Estremo + Disturbo da h2 (20 Mbps combinati)
        elif elapsed > 40:
            target_bw = "12M"
            # Ogni tanto h2 manda un burst addizionale per creare picchi
            if random.random() > 0.5:
                h2.cmd('iperf -c 10.0.0.3 -u -b 8M -t 1 &')

        # Esegui iperf client da h1 verso h3 per 1 secondo
        # Usiamo -b per la banda, -t per la durata, -u per UDP
        h1.cmd(f'iperf -c 10.0.0.3 -u -b {target_bw} -t 1')
        
        info(f"Traffic Load: {target_bw} at {int(elapsed)}s\n")

def stop_all_iperf(net):
    info("*** Uccisione processi iperf...\n")
    for host in net.hosts:
        host.cmd('killall -9 iperf')

def run():
    # Inizializza Mininet con TCLink per gestire la banda
    topo = CustomTopo()
    net = Mininet(topo=topo, link=TCLink, switch=OVSKernelSwitch)
    net.start()

    # Evento per fermare il monitor
    stop_monitor = threading.Event()
    
    # Avvia thread di monitoraggio
    monitor_thread = threading.Thread(target=monitor_traffic, args=(stop_monitor,))
    monitor_thread.start()

    try:
        # Avvia generazione traffico
        traffic_generator(net)
        
    except KeyboardInterrupt:
        info("*** Interruzione manuale...\n")
    finally:
        # Pulizia
        stop_monitor.set()
        monitor_thread.join()
        stop_all_iperf(net)
        net.stop()
        info(f"*** Dati salvati in {OUTPUT_FILE}\n")

if __name__ == '__main__':
    setLogLevel('info')
    run()
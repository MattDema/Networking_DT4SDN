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
OUTPUT_FILE = "network_data_30min.csv"
MONITORED_INTERFACE = "s1-eth2" 
SIMULATION_TIME = 1800  # 30 minuti (1800 secondi)
LINK_BW = 10            # 10 Mbps di capacità del link

class CustomTopo(Topo):
    def build(self):
        host_config = dict(inNamespace=True)
        # Link con banda limitata a 10Mbps e buffer limitato per favorire la congestione
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
    info(f"*** Avvio monitoraggio su {MONITORED_INTERFACE} per 30 minuti...\n")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("timestamp,bytes\n")
    
    prev_bytes = get_rx_bytes(MONITORED_INTERFACE)
    start_time = time.time()
    
    while not stop_event.is_set():
        # Sincronizzazione precisa al secondo
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
    
    info("*** Inizio generazione traffico dinamico (30 min)...\n")
    
    start = time.time()
    
    # Loop principale che dura SIMULATION_TIME
    while (time.time() - start) < SIMULATION_TIME:
        elapsed = time.time() - start
        
        # Logica per variare il traffico ogni 10-15 secondi
        # Scegliamo un "modo" casuale per rendere il dataset vario
        mode = random.choices(
            ['idle', 'normal', 'high', 'congestion', 'burst'], 
            weights=[10, 30, 20, 30, 10], # Probabilità (più peso su normal e congestion)
            k=1
        )[0]

        target_bw_h1 = "0M"
        burst_h2 = False

        if mode == 'idle':
            # Traffico di fondo minimo (0.5 Mbps)
            target_bw_h1 = "0.5M"
            
        elif mode == 'normal':
            # Traffico sicuro (4-6 Mbps) - Link non saturo
            bw = random.randint(4, 6)
            target_bw_h1 = f"{bw}M"
            
        elif mode == 'high':
            # Traffico al limite (8-9 Mbps) - Quasi saturo
            bw = random.uniform(8, 9.5)
            target_bw_h1 = f"{bw:.1f}M"
            
        elif mode == 'congestion':
            # Congestione (11-15 Mbps) - Perdita pacchetti sicura
            bw = random.randint(11, 15)
            target_bw_h1 = f"{bw}M"
            
        elif mode == 'burst':
            # Congestione estrema da due fonti
            target_bw_h1 = "12M"
            burst_h2 = True

        # Esecuzione comando traffico per 5 secondi (così lo stato dura un po')
        # Usiamo 'timeout' di sistema o il parametro -t di iperf
        # Qui facciamo loop brevi da 1 secondo ma manteniamo lo stato logico
        
        current_segment_duration = random.randint(5, 15)
        segment_end = time.time() + current_segment_duration
        
        info(f"[{int(elapsed)}s] Mode: {mode} -> H1 sends {target_bw_h1}" + (" + H2 BURST" if burst_h2 else "") + "\n")

        while time.time() < segment_end and (time.time() - start) < SIMULATION_TIME:
            h1.cmd(f'iperf -c 10.0.0.3 -u -b {target_bw_h1} -t 1')
            if burst_h2:
                # H2 spamma traffico addizionale per intasare tutto
                h2.cmd('iperf -c 10.0.0.3 -u -b 10M -t 1 &')
            time.sleep(0.9) # Piccola pausa per sincronizzare col loop

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
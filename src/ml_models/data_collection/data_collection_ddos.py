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
OUTPUT_FILE = "network_data_ddos.csv"
MONITORED_INTERFACE = "s1-eth2" 
SIMULATION_TIME = 1800  # 30 minuti
LINK_BW = 10            # Il collo di bottiglia è sempre 10 Mbps

class CustomTopo(Topo):
    def build(self):
        host_config = dict(inNamespace=True)
        # Link identico agli altri scenari per coerenza
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
    info(f"*** Avvio monitoraggio DDoS su {MONITORED_INTERFACE}...\n")
    
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

def run_traffic_step(host, target_ip, bandwidth, duration):
    """Esegue un singolo step di traffico"""
    host.cmd(f'iperf -c {target_ip} -u -b {bandwidth} -t {duration} &')

def traffic_generator(net):
    h1 = net.get('h1') # Botnet Master 1
    h2 = net.get('h2') # Botnet Slave 2 (per l'attacco distribuito)
    h3 = net.get('h3') # Victim

    target_ip = "10.0.0.3"
    info("*** Avvio Server Iperf sulla vittima (h3)...\n")
    h3.cmd('iperf -s -u &') 
    
    start_time = time.time()
    
    # Definiamo 3 ondate di attacco durante i 30 minuti
    # Ogni ciclo comprende: Peace -> RampUp -> PEAK -> RampDown
    
    while (time.time() - start_time) < SIMULATION_TIME:
        elapsed = time.time() - start_time
        
        # --- FASE 1: PEACE (Traffico normale pre-attacco) ---
        # Dura circa 2-3 minuti
        info(f"[{int(elapsed)}s] STATE: Peace/Normal Traffic\n")
        peace_duration = random.randint(120, 180)
        end_peace = time.time() + peace_duration
        while time.time() < end_peace:
            # Traffico basso/normale (2-4 Mbps)
            run_traffic_step(h1, target_ip, "3M", 1)
            time.sleep(1)

        # --- FASE 2: RAMP UP (Botnet Activation) ---
        # Aumentiamo il carico gradualmente da 5M a 15M in 30 secondi
        info(f"[{int(time.time()-start_time)}s] STATE: ATTACK DETECTED - RAMP UP\n")
        for bw in range(5, 16, 2): # 5M, 7M, 9M, 11M, 13M, 15M
            run_traffic_step(h1, target_ip, f"{bw}M", 5)
            # A metà ramp-up, entra anche h2 per simulare l'attacco distribuito
            if bw > 10:
                run_traffic_step(h2, target_ip, "5M", 5)
            time.sleep(5)

        # --- FASE 3: SUSTAINED PEAK (DDoS Flood) ---
        # Saturazione totale: inviamo ~25 Mbps su un link da 10 Mbps
        # Dura circa 3-4 minuti
        info(f"[{int(time.time()-start_time)}s] STATE: *** DDoS PEAK *** (Link Saturated)\n")
        peak_duration = random.randint(180, 240)
        end_peak = time.time() + peak_duration
        while time.time() < end_peak:
            # h1 spinge 15M
            run_traffic_step(h1, target_ip, "15M", 1)
            # h2 spinge 10M
            run_traffic_step(h2, target_ip, "10M", 1)
            # Totale 25M (UDP Flood) -> Packet loss massivo e buffer pieno
            time.sleep(0.9)

        # --- FASE 4: RAMP DOWN (Mitigation/Attack End) ---
        # Calo graduale in 30 secondi
        info(f"[{int(time.time()-start_time)}s] STATE: Mitigation / Ramp Down\n")
        # h2 smette subito (bloccato dal firewall simulato)
        for bw in range(15, 2, -3): # 15, 12, 9, 6, 3
            run_traffic_step(h1, target_ip, f"{bw}M", 5)
            time.sleep(5)
            
        info("*** Wave Complete. Cooling down...\n")

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

import os
import sys
import time
import threading
import requests
import numpy as np

# =============================================================================
# SETUP PATHS
# =============================================================================
# Ensure we can import modules from sibling directories
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from database.db_manager import get_db
from web_interface.app import start_web_server, get_active_switches, get_hosts, RYU_API_URL

# =============================================================================
# CONFIGURATION
# =============================================================================
COLLECTION_INTERVAL = 1  # Seconds between data collection polls

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def collect_data_periodically():
    """
    Background thread to collect and store traffic data from Ryu Controller.
    
    This function:
    1. Polls the Ryu REST API for active switches
    2. Collects Port Statistics (rx/tx bytes, packets)
    3. Collects Flow Statistics (active rules)
    4. Discovers connected Hosts
    5. Stores all data in the SQLite database
    """
    db = get_db()
    print(f"[Collector] STARTED: Polling every {COLLECTION_INTERVAL}s")

    while True:
        try:
            # 1. Get Switches
            switches = get_active_switches()
            if not switches:
                time.sleep(COLLECTION_INTERVAL)
                continue

            for dpid in switches:
                # 2. Collect Port Stats
                try:
                    url = f"{RYU_API_URL}/stats/port/{dpid}"
                    resp = requests.get(url, timeout=0.5)
                    ports = resp.json().get(str(dpid), [])
                    for port in ports:
                        db.save_port_stats(
                            dpid=dpid,
                            port_no=port.get('port_no', 0),
                            rx_packets=port.get('rx_packets', 0),
                            tx_packets=port.get('tx_packets', 0),
                            rx_bytes=port.get('rx_bytes', 0),
                            tx_bytes=port.get('tx_bytes', 0)
                        )
                except Exception:
                    pass  # Fail silently to keep collector running

                # 3. Collect Flow Stats
                try:
                    url = f"{RYU_API_URL}/stats/flow/{dpid}"
                    resp = requests.get(url, timeout=0.5)
                    flows = resp.json().get(str(dpid), [])
                    for flow in flows:
                        db.save_flow_stats(
                            dpid=dpid,
                            priority=flow.get('priority', 0),
                            match_rules=flow.get('match', {}),
                            packet_count=flow.get('packet_count', 0),
                            byte_count=flow.get('byte_count', 0),
                            actions=flow.get('actions', [])
                        )
                except Exception:
                    pass

            # 4. Save Hosts
            hosts = get_hosts()
            for host in hosts:
                db.save_host(host['mac'], host['dpid'], host['port'])

        except Exception as e:
            print(f"[Collector] ERROR: {e}")

        time.sleep(COLLECTION_INTERVAL)


def print_banner():
    """Prints the application startup banner."""
    print("\n" + "="*60)
    print("   DIGITAL TWIN ORCHESTRATOR   ")
    print("   Networking DT4SDN Project   ")
    print("="*60)
    print(f"Target Physical Twin: {RYU_API_URL}")
    print("="*60 + "\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print_banner()

    # 1. Start Collector Thread (Daemon)
    # Collects real-time stats from Mininet/Ryu
    t_col = threading.Thread(target=collect_data_periodically, daemon=True)
    t_col.start()

    # Note: Prediction loop removed as it is now handled on-demand 
    # via the Web Interface API to support multi-model switching.

    # 2. Start Web Server (Blocking)
    # Starts the Dashboard and API endpoints
    try:
        print("[Web] Starting Dashboard on port 5000...")
        start_web_server(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n[Orchestrator] Shutting down...")

import requests
import json
import time

# The Ryu REST API URL (Same as Orchestrator)
RYU_API_URL = "http://192.168.2.4:8080"  # Update IP if needed


class TrafficManager:
    def __init__(self):
        self.active_mitigations = {}  # Keep track to avoid spamming rules

    def reroute_flow(self, dpid, match_dst_ip, new_out_port):
        """
        Installs a high-priority rule to force traffic to a specific port.
        """
        # Prevent spamming the same rule every 5 seconds
        if dpid in self.active_mitigations:
            print(f"🛡 Mitigation already active on Switch {dpid}. Skipping.")
            return

        print(f"🚨 MITIGATION: Rerouting traffic for {match_dst_ip} on Switch {dpid} -> Port {new_out_port}")

        # OpenFlow Rule Definition
        flow_entry = {
            "dpid": dpid,
            "cookie": 1,
            "cookie_mask": 1,
            "table_id": 0,
            "idle_timeout": 30,  # Rule expires after 30s of silence
            "hard_timeout": 0,  # Rule stays forever otherwise
            "priority": 2000,  # Higher than normal traffic (priority 1)
            "match": {
                "dl_type": 2048,  # IPv4
                "nw_dst": match_dst_ip  # Traffic heading to h2
            },
            "actions": [
                {
                    "type": "OUTPUT",
                    "port": new_out_port
                }
            ]
        }

        # Send to Ryu
        try:
            response = requests.post(
                f"{RYU_API_URL}/stats/flowentry/add",
                data=json.dumps(flow_entry),
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                print("Reroute Rule Installed Successfully.")
                self.active_mitigations[dpid] = time.time()
            else:
                print(f"Failed to install rule: {response.text}")

        except Exception as e:
            print(f"Error communicating with controller: {e}")

    def clear_mitigations(self):
        """Optional: Remove old rules to reset the network"""
        # Logic to delete rules would go here
        self.active_mitigations.clear()
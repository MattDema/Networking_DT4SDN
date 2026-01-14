import requests
import json

SFLOW_HOST = "http://localhost:8008"


def initialize_flow_definitions():
    """
    Tells sFlow-RT to track traffic by Destination IP.
    Run this ONCE when the monitor starts.
    """
    print("🕵️ Initializing sFlow Traffic Analysis...")

    # Define a flow called 'victim_tracker'
    # keys: Group data by Destination IP ('ipdestination')
    # value: Count 'bytes'
    payload = {
        'keys': 'ipdestination',
        'value': 'bytes',
        'log': False
    }

    try:
        response = requests.put(
            f"{SFLOW_HOST}/flow/victim_tracker/json",
            json=payload
        )
        if response.status_code == 200:
            print("✅ sFlow is now tracking Victim IPs.")
        else:
            print(f"⚠️ Failed to define flow: {response.text}")
    except Exception as e:
        print(f"❌ sFlow-RT Connection Error: {e}")


# --- ADD THIS TO YOUR MAIN BLOCK ---
if __name__ == "__main__":
    initialize_flow_definitions()
    # ... rest of your monitor code ...
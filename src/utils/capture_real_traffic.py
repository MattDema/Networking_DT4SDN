#!/usr/bin/env python3
"""
capture_real_traffic.py - Capture REAL traffic from RYU REST API

This script collects traffic statistics from a running RYU controller,
computing per-second byte deltas for ML training datasets.

Usage:
    python capture_real_traffic.py --scenario normal --duration 1800 --output data.csv

Run this ON the Digital Twin VM while Mininet is running on ComNetsEmu.
"""

import requests
import time
import pandas as pd
import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional


class RYUTrafficCapture:
    """Captures real traffic statistics from RYU controller REST API"""
    
    def __init__(self, controller_ip: str = '192.168.56.101', controller_port: int = 8080):
        """
        Args:
            controller_ip: IP of the ComNetsEmu VM running RYU
            controller_port: RYU REST API port (default 8080)
        """
        self.base_url = f'http://{controller_ip}:{controller_port}'
        self.traffic_data: List[Dict] = []
        self.previous_port_bytes: Dict[str, int] = {}  # Track cumulative bytes per port
        
        print(f"RYU Controller URL: {self.base_url}")
    
    def test_connection(self) -> bool:
        """Test if RYU controller is reachable"""
        try:
            response = requests.get(f'{self.base_url}/stats/switches', timeout=5)
            if response.status_code == 200:
                switches = response.json()
                print(f"✓ Connected to RYU. Active switches: {switches}")
                return True
            else:
                print(f"✗ RYU returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to RYU at {self.base_url}")
            print("  Make sure RYU is running with: ryu-manager ryu.app.simple_switch_13 ryu.app.ofctl_rest")
            return False
        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False
    
    def get_switches(self) -> List[int]:
        """Get list of active switch DPIDs"""
        try:
            response = requests.get(f'{self.base_url}/stats/switches', timeout=2)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []
    
    def get_port_stats(self, dpid: int) -> Optional[Dict]:
        """Get port statistics for a switch"""
        try:
            url = f'{self.base_url}/stats/port/{dpid}'
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            return None
    
    def calculate_port_deltas(self, dpid: int, port_stats: Dict) -> List[Dict]:
        """
        Calculate bytes/second for each port by comparing with previous sample.
        
        Returns list of records with per-port byte deltas.
        """
        records = []
        
        ports = port_stats.get(str(dpid), [])
        
        for port in ports:
            port_no = port.get('port_no')
            
            # Skip LOCAL port (controller port)
            if port_no == 4294967294 or port_no == 'LOCAL':
                continue
            
            # Create unique key for this dpid:port
            key = f"{dpid}:{port_no}"
            
            # Current cumulative bytes (TX = bytes sent by this port)
            current_tx = port.get('tx_bytes', 0)
            current_rx = port.get('rx_bytes', 0)
            current_total = current_tx + current_rx
            
            # Calculate delta (bytes since last sample)
            previous = self.previous_port_bytes.get(key, current_total)
            delta_bytes = max(0, current_total - previous)
            
            # Update previous value for next iteration
            self.previous_port_bytes[key] = current_total
            
            records.append({
                'dpid': dpid,
                'port_no': port_no,
                'bytes_sent': delta_bytes,  # This is what the model predicts
                'tx_bytes': current_tx,
                'rx_bytes': current_rx,
            })
        
        return records
    
    def capture_traffic(self, duration_seconds: int = 1800, 
                       scenario_name: str = 'unknown',
                       sampling_rate: float = 1.0) -> List[Dict]:
        """
        Capture traffic for specified duration.
        
        Args:
            duration_seconds: How long to capture (default: 30 min)
            scenario_name: Label for this traffic scenario
            sampling_rate: Samples per second (default: 1 Hz)
        
        Returns:
            List of traffic records
        """
        print(f"\n{'='*70}")
        print(f"  CAPTURING REAL TRAFFIC: {scenario_name.upper()}")
        print(f"{'='*70}")
        print(f"  Duration: {duration_seconds}s ({duration_seconds/60:.1f} min)")
        print(f"  Sampling: {sampling_rate} Hz")
        print(f"  Output will have: timestamp, dpid, port_no, bytes_sent, scenario")
        print(f"{'='*70}\n")
        
        # Test connection first
        if not self.test_connection():
            print("\n❌ Aborting: Cannot connect to RYU controller")
            return []
        
        start_time = time.time()
        sample_count = 0
        interval = 1.0 / sampling_rate
        
        # Initial sample to establish baseline
        switches = self.get_switches()
        for dpid in switches:
            port_stats = self.get_port_stats(dpid)
            if port_stats:
                self.calculate_port_deltas(dpid, port_stats)  # Just to set baselines
        
        print(f"Started capture at {datetime.now().strftime('%H:%M:%S')}...")
        print("Press Ctrl+C to stop early\n")
        
        try:
            while (time.time() - start_time) < duration_seconds:
                sample_start = time.time()
                timestamp = sample_start - start_time
                
                # Get all switches
                switches = self.get_switches()
                
                for dpid in switches:
                    port_stats = self.get_port_stats(dpid)
                    if port_stats:
                        records = self.calculate_port_deltas(dpid, port_stats)
                        
                        for record in records:
                            self.traffic_data.append({
                                'timestamp': round(timestamp, 2),
                                'dpid': record['dpid'],
                                'port_no': record['port_no'],
                                'bytes_sent': record['bytes_sent'],
                                'scenario': scenario_name
                            })
                            sample_count += 1
                
                # Progress update every 30 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    remaining = duration_seconds - elapsed
                    # Get latest bytes for display
                    latest_bytes = self.traffic_data[-1]['bytes_sent'] if self.traffic_data else 0
                    print(f"  [{int(elapsed):4d}s] Samples: {sample_count:,} | "
                          f"Latest: {latest_bytes:,} bytes/s | ETA: {int(remaining)}s")
                
                # Maintain sampling rate
                sleep_time = interval - (time.time() - sample_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Capture interrupted by user!")
        
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"  Capture complete!")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Total samples: {len(self.traffic_data):,}")
        print(f"  Scenario: {scenario_name}")
        print(f"{'='*70}\n")
        
        return self.traffic_data
    
    def save_to_csv(self, filename: str, append: bool = False):
        """Save captured data to CSV file"""
        if not self.traffic_data:
            print("⚠️  No data to save!")
            return
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        df = pd.DataFrame(self.traffic_data)
        
        # Statistics
        print(f"\n📊 Data Statistics:")
        print(f"   Total samples: {len(df):,}")
        print(f"   Unique ports:  {df.groupby(['dpid', 'port_no']).ngroups}")
        print(f"   Min traffic:   {df['bytes_sent'].min():,.0f} bytes/s")
        print(f"   Max traffic:   {df['bytes_sent'].max():,.0f} bytes/s")
        print(f"   Mean traffic:  {df['bytes_sent'].mean():,.0f} bytes/s")
        print(f"   Std dev:       {df['bytes_sent'].std():,.0f} bytes/s")
        
        # Save (append or overwrite)
        if append and os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
            print(f"\n✓ Appended to: {filename}")
        else:
            df.to_csv(filename, index=False)
            print(f"\n✓ Saved to: {filename}")
        
        return df
    
    def reset(self):
        """Reset captured data for next scenario"""
        self.traffic_data = []
        self.previous_port_bytes = {}


def main():
    parser = argparse.ArgumentParser(
        description='Capture real traffic from RYU controller for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture normal traffic for 30 minutes
  python capture_real_traffic.py --scenario normal --duration 1800
  
  # Capture from remote RYU
  python capture_real_traffic.py --scenario ddos --controller-ip 192.168.56.101
  
  # Quick test (1 minute)
  python capture_real_traffic.py --scenario test --duration 60 --output test.csv
        """
    )
    
    parser.add_argument('--scenario', type=str, required=True,
                       choices=['normal', 'burst', 'congestion', 'ddos', 'mixed', 'test'],
                       help='Traffic scenario name for labeling')
    parser.add_argument('--duration', type=int, default=1800,
                       help='Capture duration in seconds (default: 1800 = 30 min)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV path (default: auto-generated)')
    parser.add_argument('--controller-ip', type=str, default='192.168.56.101',
                       help='RYU controller IP (default: 192.168.56.101)')
    parser.add_argument('--controller-port', type=int, default=8080,
                       help='RYU REST API port (default: 8080)')
    parser.add_argument('--sampling-rate', type=float, default=1.0,
                       help='Samples per second (default: 1.0)')
    parser.add_argument('--append', action='store_true',
                       help='Append to existing CSV instead of overwriting')
    
    args = parser.parse_args()
    
    # Auto-generate output filename
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'data/real_traffic/{args.scenario}_{timestamp}.csv'
    
    # Capture
    capturer = RYUTrafficCapture(args.controller_ip, args.controller_port)
    capturer.capture_traffic(
        duration_seconds=args.duration,
        scenario_name=args.scenario,
        sampling_rate=args.sampling_rate
    )
    
    # Save
    capturer.save_to_csv(args.output, append=args.append)


if __name__ == '__main__':
    main()

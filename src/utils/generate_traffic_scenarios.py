#!/usr/bin/env python3
"""
generate_traffic_scenarios.py - Generate different traffic scenarios in Mininet

This script runs on the ComNetsEmu VM and generates various traffic patterns
while capture_real_traffic.py (on Digital Twin VM) records the data.

Usage:
    sudo python generate_traffic_scenarios.py --scenario normal --duration 1800

Requirements:
    - Mininet running with topology (custom_topo.py)
    - iperf installed (comes with ComNetsEmu)
    - hping3 or nping for DDoS (optional: sudo apt install hping3)
"""

import time
import argparse
import subprocess
import random
import os
import sys
from typing import List


class TrafficGenerator:
    """Generate various traffic patterns using iperf/hping3"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.host_ips = [
            '10.0.0.1',  # h1
            '10.0.0.2',  # h2
            '10.0.0.3',  # h3
        ]
        
    def _run_cmd(self, host_idx: int, cmd: str, background: bool = True):
        """Run command on a Mininet host via m(host) wrapper or direct"""
        # In ComNetsEmu, we use 'mx h1 <cmd>' or direct subprocess
        # For standalone, we'll use subprocess with the expectation
        # that Mininet CLI is handling host namespaces
        
        full_cmd = f"mx h{host_idx + 1} {cmd}"
        
        if background:
            proc = subprocess.Popen(
                full_cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.processes.append(proc)
            return proc
        else:
            return subprocess.run(full_cmd, shell=True, capture_output=True)
    
    def _cleanup(self):
        """Kill all spawned processes"""
        for proc in self.processes:
            try:
                proc.terminate()
            except:
                pass
        
        # Kill all iperf and hping3 processes
        subprocess.run("sudo killall -9 iperf iperf3 hping3 2>/dev/null", 
                      shell=True, capture_output=True)
        self.processes = []
    
    def _start_iperf_servers(self):
        """Start iperf servers on all hosts"""
        print("  Starting iperf servers on all hosts...")
        for i in range(len(self.host_ips)):
            self._run_cmd(i, "iperf -s -u -p 5001 &", background=True)
        time.sleep(2)
    
    def generate_normal(self, duration: int = 1800):
        """
        Normal traffic: Light background traffic (~100K - 1M bps)
        Simulates typical office/home network usage
        """
        print(f"\n{'='*60}")
        print(f"  GENERATING NORMAL TRAFFIC ({duration}s)")
        print(f"{'='*60}")
        print(f"  Pattern: Light background traffic, 100K-1M bps")
        print(f"  Method:  iperf UDP from h1 to h3")
        print(f"{'='*60}\n")
        
        self._start_iperf_servers()
        
        start = time.time()
        segment = 0
        
        try:
            while (time.time() - start) < duration:
                elapsed = int(time.time() - start)
                segment += 1
                
                # Random bandwidth: 100K to 1M
                bandwidth = random.randint(100, 1000)
                segment_duration = random.randint(10, 30)
                
                print(f"  [{elapsed:4d}s] Segment {segment}: {bandwidth}K bps for {segment_duration}s")
                
                # h1 -> h3 traffic
                self._run_cmd(0, f"iperf -c 10.0.0.3 -u -b {bandwidth}K -t {segment_duration}")
                
                time.sleep(segment_duration)
        
        except KeyboardInterrupt:
            print("\n  Interrupted!")
        finally:
            self._cleanup()
        
        print(f"\n✓ Normal traffic generation complete ({int(time.time() - start)}s)")
    
    def generate_burst(self, duration: int = 1800):
        """
        Burst traffic: Normal baseline with periodic 10 Mbps spikes
        Simulates video calls, large file transfers
        """
        print(f"\n{'='*60}")
        print(f"  GENERATING BURST TRAFFIC ({duration}s)")
        print(f"{'='*60}")
        print(f"  Pattern: 10s quiet (500K) -> 10s burst (10M)")
        print(f"  Method:  iperf UDP alternating rates")
        print(f"{'='*60}\n")
        
        self._start_iperf_servers()
        
        start = time.time()
        is_burst = False
        
        try:
            while (time.time() - start) < duration:
                elapsed = int(time.time() - start)
                
                if is_burst:
                    print(f"  [{elapsed:4d}s] 🔥 BURST: 10 Mbps")
                    self._run_cmd(0, "iperf -c 10.0.0.3 -u -b 10M -t 10")
                    time.sleep(10)
                else:
                    print(f"  [{elapsed:4d}s] 💤 Quiet: 500 Kbps")
                    self._run_cmd(0, "iperf -c 10.0.0.3 -u -b 500K -t 15")
                    time.sleep(15)
                
                is_burst = not is_burst
        
        except KeyboardInterrupt:
            print("\n  Interrupted!")
        finally:
            self._cleanup()
        
        print(f"\n✓ Burst traffic generation complete ({int(time.time() - start)}s)")
    
    def generate_congestion(self, duration: int = 1800):
        """
        Congestion: All hosts sending to each other simultaneously
        Simulates network overload, backup operations
        """
        print(f"\n{'='*60}")
        print(f"  GENERATING CONGESTION TRAFFIC ({duration}s)")
        print(f"{'='*60}")
        print(f"  Pattern: All hosts sending 3-5 Mbps simultaneously")
        print(f"  Method:  Multiple iperf flows competing")
        print(f"{'='*60}\n")
        
        self._start_iperf_servers()
        
        start = time.time()
        
        try:
            while (time.time() - start) < duration:
                elapsed = int(time.time() - start)
                
                # Random bandwidth per flow
                bw1 = random.randint(3, 5)
                bw2 = random.randint(2, 4)
                segment_duration = 20
                
                print(f"  [{elapsed:4d}s] Congestion: h1->{bw1}M, h2->{bw2}M")
                
                # h1 -> h3
                self._run_cmd(0, f"iperf -c 10.0.0.3 -u -b {bw1}M -t {segment_duration}")
                # h2 -> h3 (additional load)
                self._run_cmd(1, f"iperf -c 10.0.0.3 -u -b {bw2}M -t {segment_duration}")
                # h1 -> h2 (cross traffic)
                self._run_cmd(0, f"iperf -c 10.0.0.2 -u -b 2M -t {segment_duration} -p 5002")
                
                time.sleep(segment_duration)
        
        except KeyboardInterrupt:
            print("\n  Interrupted!")
        finally:
            self._cleanup()
        
        print(f"\n✓ Congestion traffic generation complete ({int(time.time() - start)}s)")
    
    def generate_ddos(self, duration: int = 1800):
        """
        DDoS simulation: Periodic flood attacks
        Uses hping3 for SYN flood or high-rate UDP
        """
        print(f"\n{'='*60}")
        print(f"  GENERATING DDOS TRAFFIC ({duration}s)")
        print(f"{'='*60}")
        print(f"  Pattern: 60s normal -> 30s attack (flood)")
        print(f"  Method:  hping3 SYN flood or high-rate iperf")
        print(f"{'='*60}\n")
        
        # Check if hping3 is available
        has_hping3 = subprocess.run("which hping3", shell=True, capture_output=True).returncode == 0
        
        if has_hping3:
            print("  Using hping3 for attack simulation")
        else:
            print("  ⚠️  hping3 not found, using high-rate iperf instead")
            print("     Install with: sudo apt install hping3")
        
        self._start_iperf_servers()
        
        start = time.time()
        attack_num = 0
        
        try:
            while (time.time() - start) < duration:
                elapsed = int(time.time() - start)
                
                # Normal phase (60s)
                print(f"  [{elapsed:4d}s] 💤 Normal phase (60s)")
                self._run_cmd(0, "iperf -c 10.0.0.3 -u -b 500K -t 60")
                time.sleep(60)
                
                if (time.time() - start) >= duration:
                    break
                
                # Attack phase (30s)
                attack_num += 1
                elapsed = int(time.time() - start)
                print(f"  [{elapsed:4d}s] 🔥 ATTACK #{attack_num} (30s)")
                
                if has_hping3:
                    # SYN flood using hping3
                    self._run_cmd(0, f"hping3 -S -p 80 --flood 10.0.0.3 &")
                    self._run_cmd(1, f"hping3 -S -p 80 --flood 10.0.0.3 &")
                else:
                    # High-rate UDP flood
                    self._run_cmd(0, "iperf -c 10.0.0.3 -u -b 50M -t 30")
                    self._run_cmd(1, "iperf -c 10.0.0.3 -u -b 50M -t 30")
                
                time.sleep(30)
                
                # Stop attack
                if has_hping3:
                    subprocess.run("sudo killall hping3 2>/dev/null", shell=True)
        
        except KeyboardInterrupt:
            print("\n  Interrupted!")
        finally:
            self._cleanup()
        
        print(f"\n✓ DDoS traffic generation complete ({int(time.time() - start)}s)")
    
    def generate_mixed(self, duration: int = 1800):
        """
        Mixed traffic: Random combination of all patterns
        Most realistic scenario for training
        """
        print(f"\n{'='*60}")
        print(f"  GENERATING MIXED TRAFFIC ({duration}s)")
        print(f"{'='*60}")
        print(f"  Pattern: Random mix of normal, burst, congestion")
        print(f"  Method:  Weighted random selection every 30-60s")
        print(f"{'='*60}\n")
        
        self._start_iperf_servers()
        
        start = time.time()
        
        try:
            while (time.time() - start) < duration:
                elapsed = int(time.time() - start)
                
                # Random pattern selection
                pattern = random.choices(
                    ['normal', 'burst', 'congestion'],
                    weights=[40, 30, 30],
                    k=1
                )[0]
                
                segment_duration = random.randint(30, 60)
                
                if pattern == 'normal':
                    bw = random.randint(200, 800)
                    print(f"  [{elapsed:4d}s] Normal ({bw}K) for {segment_duration}s")
                    self._run_cmd(0, f"iperf -c 10.0.0.3 -u -b {bw}K -t {segment_duration}")
                
                elif pattern == 'burst':
                    print(f"  [{elapsed:4d}s] Burst (10M) for {segment_duration}s")
                    self._run_cmd(0, f"iperf -c 10.0.0.3 -u -b 10M -t {segment_duration}")
                
                elif pattern == 'congestion':
                    print(f"  [{elapsed:4d}s] Congestion (multi-flow) for {segment_duration}s")
                    self._run_cmd(0, f"iperf -c 10.0.0.3 -u -b 4M -t {segment_duration}")
                    self._run_cmd(1, f"iperf -c 10.0.0.3 -u -b 3M -t {segment_duration}")
                
                time.sleep(segment_duration)
        
        except KeyboardInterrupt:
            print("\n  Interrupted!")
        finally:
            self._cleanup()
        
        print(f"\n✓ Mixed traffic generation complete ({int(time.time() - start)}s)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate traffic scenarios in Mininet for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate normal traffic for 30 minutes
  sudo python generate_traffic_scenarios.py --scenario normal --duration 1800
  
  # Generate DDoS pattern for 10 minutes
  sudo python generate_traffic_scenarios.py --scenario ddos --duration 600
  
  # Quick test (1 minute)
  sudo python generate_traffic_scenarios.py --scenario test --duration 60
        """
    )
    
    parser.add_argument('--scenario', type=str, required=True,
                       choices=['normal', 'burst', 'congestion', 'ddos', 'mixed', 'test'],
                       help='Traffic pattern to generate')
    parser.add_argument('--duration', type=int, default=1800,
                       help='Duration in seconds (default: 1800 = 30 min)')
    
    args = parser.parse_args()
    
    # Check if running as root (needed for Mininet/hping3)
    if os.geteuid() != 0:
        print("⚠️  Warning: This script may need sudo for full functionality")
        print("    Run with: sudo python generate_traffic_scenarios.py ...")
    
    generator = TrafficGenerator()
    
    if args.scenario == 'normal':
        generator.generate_normal(args.duration)
    elif args.scenario == 'burst':
        generator.generate_burst(args.duration)
    elif args.scenario == 'congestion':
        generator.generate_congestion(args.duration)
    elif args.scenario == 'ddos':
        generator.generate_ddos(args.duration)
    elif args.scenario == 'mixed':
        generator.generate_mixed(args.duration)
    elif args.scenario == 'test':
        print("Running quick test (mixed pattern)...")
        generator.generate_mixed(args.duration)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
run_overnight_collection.py - Batch data collection for all scenarios

This script runs ALL traffic scenarios sequentially, collecting training data
for ML models. Designed to run overnight (~2.5-7.5 hours depending on settings).

IMPORTANT: Run this on the ComNetsEmu VM (where Mininet is running).
The capture_real_traffic.py should be running on the Digital Twin VM.

Usage:
    # Full overnight run (2.5 hours)
    sudo python run_overnight_collection.py
    
    # Quick test (5 minutes total)
    sudo python run_overnight_collection.py --duration-per-scenario 60
    
    # Multiple rounds (~7.5 hours)
    sudo python run_overnight_collection.py --rounds 3
"""

import subprocess
import time
import argparse
import os
import sys
from datetime import datetime, timedelta


# Configuration
SCENARIOS = ['normal', 'burst', 'congestion', 'ddos', 'mixed']
DEFAULT_DURATION_PER_SCENARIO = 1800  # 30 minutes each

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_banner():
    """Print startup banner"""
    print(f"""
{Colors.BLUE}{'='*70}
 ____  _   _ _   _   _____           __  __ _        ____       _        
|  _ \\| \\ | | \\ | | |_   _| __ __ _ / _|/ _(_) ___  |  _ \\  ___| |_ __ _ 
| |_) |  \\| |  \\| |   | || '__/ _` | |_| |_| |/ __| | | | |/ _ \\ __/ _` |
|  _ <| |\\  | |\\  |   | || | | (_| |  _|  _| | (__  | |_| |  __/ || (_| |
|_| \\_\\_| \\_|_| \\_|   |_||_|  \\__,_|_| |_| |_|\\___| |____/ \\___|\\__\\__,_|
                                                                          
              OVERNIGHT DATA COLLECTION FOR ML TRAINING
{'='*70}{Colors.END}
""")


def estimate_duration(num_scenarios: int, duration_per: int, rounds: int) -> str:
    """Calculate estimated total duration"""
    total_seconds = num_scenarios * duration_per * rounds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    
    end_time = datetime.now() + timedelta(seconds=total_seconds)
    
    return f"{hours}h {minutes}m (finishes ~{end_time.strftime('%H:%M')})"


def run_scenario(scenario: str, duration: int, round_num: int) -> bool:
    """
    Run traffic generation for a single scenario.
    Returns True if successful, False otherwise.
    """
    print(f"\n{Colors.YELLOW}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  Round {round_num} | Scenario: {scenario.upper()}{Colors.END}")
    print(f"  Duration: {duration}s ({duration/60:.0f} min)")
    print(f"  Started:  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{Colors.YELLOW}{'='*60}{Colors.END}\n")
    
    try:
        # Run the traffic generation script
        cmd = [
            sys.executable,  # Use same Python interpreter
            'generate_traffic_scenarios.py',
            '--scenario', scenario,
            '--duration', str(duration)
        ]
        
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print(f"{Colors.GREEN}✓ Scenario '{scenario}' completed successfully{Colors.END}")
            return True
        else:
            print(f"{Colors.RED}✗ Scenario '{scenario}' failed with code {result.returncode}{Colors.END}")
            return False
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Interrupted by user{Colors.END}")
        raise
    except Exception as e:
        print(f"{Colors.RED}✗ Error running scenario '{scenario}': {e}{Colors.END}")
        return False


def run_all_scenarios(duration_per_scenario: int, rounds: int):
    """Run all scenarios for specified number of rounds"""
    
    print_banner()
    
    total_scenarios = len(SCENARIOS) * rounds
    estimated = estimate_duration(len(SCENARIOS), duration_per_scenario, rounds)
    
    print(f"{Colors.BLUE}Configuration:{Colors.END}")
    print(f"  Scenarios:      {', '.join(SCENARIOS)}")
    print(f"  Duration each:  {duration_per_scenario}s ({duration_per_scenario/60:.0f} min)")
    print(f"  Rounds:         {rounds}")
    print(f"  Total time:     {estimated}")
    print()
    
    # Confirmation
    print(f"{Colors.YELLOW}⚠️  IMPORTANT: Make sure capture_real_traffic.py is running on Digital Twin VM!{Colors.END}")
    print()
    print("Starting in 10 seconds... (Ctrl+C to cancel)")
    
    try:
        for i in range(10, 0, -1):
            print(f"  {i}...", end=' ', flush=True)
            time.sleep(1)
        print("\n")
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return
    
    # Run all scenarios
    start_time = time.time()
    completed = 0
    failed = 0
    
    try:
        for round_num in range(1, rounds + 1):
            print(f"\n{Colors.BLUE}{'#'*70}")
            print(f"#  ROUND {round_num} of {rounds}")
            print(f"{'#'*70}{Colors.END}")
            
            for scenario in SCENARIOS:
                try:
                    success = run_scenario(scenario, duration_per_scenario, round_num)
                    if success:
                        completed += 1
                    else:
                        failed += 1
                    
                    # Brief pause between scenarios
                    if scenario != SCENARIOS[-1] or round_num != rounds:
                        print("\n  Pausing 5s before next scenario...")
                        time.sleep(5)
                
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"{Colors.RED}Error in {scenario}: {e}{Colors.END}")
                    failed += 1
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Collection interrupted by user{Colors.END}")
    
    # Summary
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print(f"\n{Colors.GREEN}{'='*70}")
    print(f"  DATA COLLECTION COMPLETE")
    print(f"{'='*70}{Colors.END}")
    print(f"  Completed:  {completed}/{total_scenarios} scenarios")
    print(f"  Failed:     {failed}")
    print(f"  Duration:   {hours}h {minutes}m")
    print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"  📁 Data should be in: data/real_traffic/")
    print(f"  📊 Merge with: cat data/real_traffic/*.csv > data/training/real_combined.csv")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run overnight data collection for all traffic scenarios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full overnight collection (2.5 hours)
  sudo python run_overnight_collection.py
  
  # Quick test (5 minutes total)
  sudo python run_overnight_collection.py --duration-per-scenario 60
  
  # Extended collection (7.5 hours, 3 rounds)
  sudo python run_overnight_collection.py --rounds 3
  
  # Custom duration (15 min per scenario, 2 rounds)
  sudo python run_overnight_collection.py --duration-per-scenario 900 --rounds 2

IMPORTANT:
  This script generates traffic on ComNetsEmu VM.
  Make sure to run capture_real_traffic.py on the Digital Twin VM
  BEFORE starting this script!
        """
    )
    
    parser.add_argument('--duration-per-scenario', type=int, 
                       default=DEFAULT_DURATION_PER_SCENARIO,
                       help=f'Duration per scenario in seconds (default: {DEFAULT_DURATION_PER_SCENARIO})')
    parser.add_argument('--rounds', type=int, default=1,
                       help='Number of rounds to repeat all scenarios (default: 1)')
    parser.add_argument('--scenarios', type=str, nargs='+',
                       choices=SCENARIOS + ['all'],
                       default=['all'],
                       help='Specific scenarios to run (default: all)')
    
    args = parser.parse_args()
    
    # Filter scenarios if specified
    global SCENARIOS
    if 'all' not in args.scenarios:
        SCENARIOS = [s for s in args.scenarios if s in SCENARIOS]
    
    # Check if running as root
    if os.geteuid() != 0:
        print(f"{Colors.YELLOW}⚠️  Warning: Running without sudo - some features may not work{Colors.END}")
    
    run_all_scenarios(args.duration_per_scenario, args.rounds)


if __name__ == '__main__':
    main()

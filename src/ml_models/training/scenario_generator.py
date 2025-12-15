"""
Generate synthetic traffic patterns for training
Models different network scenarios: normal, DDoS, congestion, burst
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import random

class TrafficScenarioGenerator:
    """
    Generate realistic SDN traffic patterns for multiple scenarios
    """
    
    def __init__(self, duration_seconds: int = 600, sampling_rate: float = 1.0):
        """
        Args:
            duration_seconds: How long to simulate (e.g., 600 = 10 minutes)
            sampling_rate: Samples per second (1.0 = every second)
        """
        self.duration = duration_seconds
        self.sampling_rate = sampling_rate
        self.timestamps = np.arange(0, duration_seconds, 1/sampling_rate)
    
    def generate_normal_traffic(self, base_rate: int = 2000, 
                                variation: float = 0.2) -> np.ndarray:
        """
        Normal network operation
        - Stable baseline with small random fluctuations
        - Occasional small peaks (business hours pattern)
        
        Args:
            base_rate: Average bytes per second
            variation: Random variation (0.2 = ±20%)
        
        Returns:
            Array of traffic values (bytes/second)
        """
        traffic = []
        
        for t in self.timestamps:
            # Base load with noise
            noise = np.random.normal(0, base_rate * variation)
            
            # Diurnal pattern (higher during "work hours")
            hour = (t / 3600) % 24  # Hour of day
            if 8 <= hour <= 18:  # Work hours
                multiplier = 1.2
            else:
                multiplier = 0.8
            
            value = base_rate * multiplier + noise
            traffic.append(max(0, value))  # No negative traffic
        
        return np.array(traffic)
    
    def generate_burst_traffic(self, base_rate: int = 2000,
                               num_bursts: int = 5,
                               burst_multiplier: float = 5.0) -> np.ndarray:
        """
        Sudden traffic bursts (e.g., video conference starts, file transfer)
        - Normal baseline
        - Random sudden spikes that last 10-60 seconds
        
        Args:
            base_rate: Normal traffic rate
            num_bursts: Number of burst events
            burst_multiplier: How much traffic increases during burst
        """
        traffic = self.generate_normal_traffic(base_rate, variation=0.1)
        
        # Add random bursts
        for _ in range(num_bursts):
            burst_start = random.randint(0, len(traffic) - 60)
            burst_duration = random.randint(10, 60)  # 10-60 seconds
            
            for i in range(burst_start, min(burst_start + burst_duration, len(traffic))):
                traffic[i] *= burst_multiplier
        
        return traffic
    
    def generate_congestion_traffic(self, base_rate: int = 2000,
                                    peak_multiplier: float = 8.0) -> np.ndarray:
        """
        Gradual congestion buildup
        - Starts normal
        - Gradually increases to peak
        - Stays high for a while
        - Gradually decreases
        
        Simulates: End-of-day backup, data replication, etc.
        """
        traffic = []
        
        # Phase 1: Normal (0-20%)
        phase1_end = int(len(self.timestamps) * 0.2)
        traffic.extend(self.generate_normal_traffic(base_rate)[:phase1_end])
        
        # Phase 2: Ramp up (20-40%)
        phase2_end = int(len(self.timestamps) * 0.4)
        ramp_up = np.linspace(base_rate, base_rate * peak_multiplier, 
                              phase2_end - phase1_end)
        traffic.extend(ramp_up)
        
        # Phase 3: Peak congestion (40-70%)
        phase3_end = int(len(self.timestamps) * 0.7)
        peak_traffic = self.generate_normal_traffic(base_rate * peak_multiplier, 
                                                     variation=0.15)
        traffic.extend(peak_traffic[:phase3_end - phase2_end])
        
        # Phase 4: Ramp down (70-90%)
        phase4_end = int(len(self.timestamps) * 0.9)
        ramp_down = np.linspace(base_rate * peak_multiplier, base_rate,
                                phase4_end - phase3_end)
        traffic.extend(ramp_down)
        
        # Phase 5: Normal (90-100%)
        remaining = len(self.timestamps) - len(traffic)
        traffic.extend(self.generate_normal_traffic(base_rate)[:remaining])
        
        return np.array(traffic)
    
    def generate_ddos_traffic(self, base_rate: int = 2000,
                             attack_multiplier: float = 20.0,
                             attack_duration: int = 120) -> np.ndarray:
        """
        DDoS attack simulation
        - Normal traffic
        - Sudden MASSIVE spike (attack starts)
        - Sustained high traffic
        - Sudden drop (attack mitigated)
        
        Args:
            base_rate: Normal traffic
            attack_multiplier: How much traffic increases (20x = severe attack)
            attack_duration: How long attack lasts (seconds)
        """
        traffic = self.generate_normal_traffic(base_rate, variation=0.1)
        
        # Random attack start time (not too early/late)
        attack_start = random.randint(int(len(traffic) * 0.2), 
                                      int(len(traffic) * 0.6))
        attack_end = min(attack_start + attack_duration, len(traffic))
        
        # DDoS attack characteristics:
        # 1. Sudden spike at start
        # 2. Sustained very high traffic
        # 3. Sudden drop when mitigated
        
        for i in range(attack_start, attack_end):
            # Very high traffic with some randomness
            attack_traffic = base_rate * attack_multiplier
            noise = np.random.normal(0, attack_traffic * 0.1)
            traffic[i] = attack_traffic + noise
        
        return traffic
    
    def generate_mixed_traffic(self, base_rate: int = 2000) -> np.ndarray:
        """
        Realistic mixed traffic: normal + occasional bursts + brief congestion
        Most realistic scenario for training
        """
        # Start with normal traffic
        traffic = self.generate_normal_traffic(base_rate, variation=0.15)
        
        # Add 2-3 bursts
        num_bursts = random.randint(2, 3)
        for _ in range(num_bursts):
            burst_start = random.randint(0, len(traffic) - 60)
            burst_duration = random.randint(15, 45)
            burst_multiplier = random.uniform(3.0, 6.0)
            
            for i in range(burst_start, min(burst_start + burst_duration, len(traffic))):
                traffic[i] *= burst_multiplier
        
        # Add brief congestion period
        congestion_start = random.randint(int(len(traffic) * 0.5), 
                                         int(len(traffic) * 0.7))
        congestion_duration = random.randint(60, 120)
        
        for i in range(congestion_start, 
                      min(congestion_start + congestion_duration, len(traffic))):
            traffic[i] *= random.uniform(2.0, 4.0)
        
        return traffic
    
    def generate_all_scenarios(self, output_csv: str = 'data/training/traffic_scenarios.csv'):
        """
        Generate training dataset with all scenarios
        
        Creates CSV with columns:
        - timestamp: Time in seconds
        - bytes_sent: Traffic volume
        - scenario: Label (normal/burst/congestion/ddos/mixed)
        """
        all_data = []
        
        scenarios = {
            'normal': self.generate_normal_traffic(),
            'burst': self.generate_burst_traffic(),
            'congestion': self.generate_congestion_traffic(),
            'ddos': self.generate_ddos_traffic(),
            'mixed': self.generate_mixed_traffic()
        }
        
        for scenario_name, traffic_data in scenarios.items():
            for t, bytes_val in zip(self.timestamps, traffic_data):
                all_data.append({
                    'timestamp': t,
                    'bytes_sent': int(bytes_val),
                    'scenario': scenario_name
                })
        
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"✓ Generated {len(all_data)} samples across {len(scenarios)} scenarios")
        print(f"  Saved to: {output_csv}")
        
        return df


def main():
    """Generate training data"""
    print("=" * 60)
    print("Traffic Scenario Generator")
    print("=" * 60)
    
    # Generate 10 minutes of traffic for each scenario
    generator = TrafficScenarioGenerator(duration_seconds=600, sampling_rate=1.0)
    
    df = generator.generate_all_scenarios('data/training/traffic_scenarios.csv')
    
    # Print statistics
    print("\nDataset Statistics:")
    print(df.groupby('scenario')['bytes_sent'].describe())
    
    # Plot examples (optional)
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, (scenario, data) in enumerate(df.groupby('scenario')):
            axes[idx].plot(data['timestamp'], data['bytes_sent'])
            axes[idx].set_title(f'{scenario.capitalize()} Traffic')
            axes[idx].set_xlabel('Time (s)')
            axes[idx].set_ylabel('Bytes/s')
            axes[idx].grid(True)
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig('docs/traffic_scenarios.png')
        print(f"\n✓ Visualization saved to docs/traffic_scenarios.png")
    except ImportError:
        print("\nMatplotlib not available, skipping visualization")


if __name__ == '__main__':
    main()

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
    
    # """ def generate_normal_traffic(self, base_rate: int = 2000, 
    #                             variation: float = 0.3) -> np.ndarray:  # Increase from 0.2 to 0.3
    #     # 
    #     # Normal network operation
    #     # - Stable baseline with small random fluctuations
    #     # - Occasional small peaks (business hours pattern)
        
    #     # Args:
    #     #     base_rate: Average bytes per second
    #     #     variation: Random variation (0.2 = ±20%)
        
    #     # Returns:
    #     #     Array of traffic values (bytes/second)
        
    #     traffic = []
    
    #     for t in self.timestamps:
    #         # More noise
    #         noise = np.random.normal(0, base_rate * variation)
        
    #         # Add random micro-bursts (realistic)
    #         if np.random.random() < 0.05:  # 5% chance of small spike
    #             noise += base_rate * 0.5
        
    #         # Diurnal pattern with more variation
    #         hour = (t / 3600) % 24
    #         if 8 <= hour <= 18:
    #             multiplier = np.random.uniform(1.1, 1.3)  # Random variation
    #         else:
    #             multiplier = np.random.uniform(0.7, 0.9)
        
    #         value = base_rate * multiplier + noise
    #         traffic.append(max(0, value))

    #     return np.array(traffic) """

    def generate_normal_traffic(self, base_rate: int = 2000, variation: float = 0.3) -> np.ndarray:
        """
        Realistic normal traffic with:
        - Daily patterns (morning/evening peaks)
        - Weekly patterns (weekday vs weekend)
        - Random walk trends
        - Micro-bursts
        """
        traffic = []
    
        # Add slow trending (traffic grows/shrinks over hours)
        trend = 0
        trend_change_rate = 0.0001
    
        for t in self.timestamps:
            # Time-based patterns
            hour = (t / 3600) % 24
            day = int(t / 86400) % 7
        
            # Daily pattern (business hours)
            if 6 <= hour <= 9:  # Morning ramp-up
                time_multiplier = 0.7 + (hour - 6) * 0.15
            elif 9 <= hour <= 17:  # Peak hours
                time_multiplier = 1.2 + np.random.uniform(-0.1, 0.1)
            elif 17 <= hour <= 20:  # Evening decline
                time_multiplier = 1.1 - (hour - 17) * 0.1
            else:  # Night
                time_multiplier = 0.5 + np.random.uniform(-0.1, 0.1)
        
            # Weekly pattern (weekday vs weekend)
            if day >= 5:  # Weekend
                time_multiplier *= 0.6
        
            # Add trending behavior
            trend += np.random.uniform(-trend_change_rate, trend_change_rate)
            trend = np.clip(trend, -0.3, 0.3)
        
            # Base traffic with all patterns
            base = base_rate * (1 + trend) * time_multiplier
        
            # Add noise
            noise = np.random.normal(0, base * variation)
        
            # Random micro-bursts (5% chance)
            if np.random.random() < 0.05:
                noise += base * np.random.uniform(0.3, 0.8)
        
            value = base + noise
            traffic.append(max(100, value))  # Minimum 100 bytes
    
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
    
    # def generate_ddos_traffic(self, base_rate: int = 2000,
    #                      attack_multiplier: float = 20.0,
    #                      attack_duration: int = 120) -> np.ndarray:
    #     """
    #     DDoS attack simulation
    #     - Normal traffic
    #     - Sudden MASSIVE spike (attack starts)
    #     - Sustained high traffic
    #     - Sudden drop (attack mitigated)
        
    #     Args:
    #         base_rate: Normal traffic
    #         attack_multiplier: How much traffic increases (20x = severe attack)
    #         attack_duration: How long attack lasts (seconds)
    #     """
    #     traffic = self.generate_normal_traffic(base_rate, variation=0.15)
        
    #     # # Random attack start time (not too early/late)
    #     # attack_start = random.randint(int(len(traffic) * 0.2), 
    #     #                               int(len(traffic) * 0.6))
    #     # attack_end = min(attack_start + attack_duration, len(traffic))
        
    #     # DDoS attack characteristics:
    #     # 1. Sudden spike at start
    #     # 2. Sustained very high traffic
    #     # 3. Sudden drop when mitigated
        
    #     # Multiple attack waves (more realistic)
    #     num_attacks = np.random.randint(2, 4)  # 2-3 attack waves
    
    #     for _ in range(num_attacks):
    #         attack_start = np.random.randint(int(len(traffic) * 0.2), 
    #                                     int(len(traffic) * 0.7))
    #         attack_end = min(attack_start + attack_duration + np.random.randint(-20, 20), 
    #                     len(traffic))
        
    #         for i in range(attack_start, attack_end):
    #             # Attack traffic varies (ramping up/down)
    #             progress = (i - attack_start) / (attack_end - attack_start)
                
    #             # Ramp up, plateau, ramp down
    #             if progress < 0.1:
    #                 intensity = progress * 10  # Ramp up
    #             elif progress > 0.9:
    #                 intensity = (1 - progress) * 10  # Ramp down
    #             else:
    #                 intensity = 1.0  # Plateau
            
    #             attack_traffic = base_rate * attack_multiplier * intensity
    #             noise = np.random.normal(0, attack_traffic * 0.1)
    #             traffic[i] = attack_traffic + noise
    
    #     return traffic
    
    def generate_ddos_traffic(self, base_rate: int = 2000,
                         attack_multiplier: float = 15.0,
                         attack_duration: int = 180) -> np.ndarray:
        """
        Realistic DDoS with:
        - Gradual ramp-up (botnet activation)
        - Sustained peak
        - Gradual ramp-down (mitigation)
        - Multiple attack waves
        """
        traffic = self.generate_normal_traffic(base_rate, variation=0.15)
    
        # Generate 2-3 attack waves
        num_attacks = np.random.randint(2, 4)
    
        for attack_num in range(num_attacks):
            # Random attack timing
            attack_start = np.random.randint(
                int(len(traffic) * 0.2) + attack_num * 2000,
                int(len(traffic) * 0.7) - attack_duration
            )
        
            # Attack phases
            ramp_up_duration = 30  # 30 seconds to ramp up
            ramp_down_duration = 20  # 20 seconds to ramp down
            plateau_duration = attack_duration - ramp_up_duration - ramp_down_duration
        
            # Phase 1: Ramp up (gradual botnet activation)
            for i in range(ramp_up_duration):
                idx = attack_start + i
                if idx < len(traffic):
                    progress = i / ramp_up_duration  # 0 to 1
                    intensity = progress ** 2  # Quadratic ramp
                    attack_level = base_rate * attack_multiplier * intensity
                    noise = np.random.normal(0, attack_level * 0.1)
                    traffic[idx] = traffic[idx] + attack_level + noise
        
            # Phase 2: Sustained attack (plateau)
            for i in range(plateau_duration):
                idx = attack_start + ramp_up_duration + i
                if idx < len(traffic):
                    # Fluctuating high traffic
                    intensity = 0.9 + np.random.uniform(-0.1, 0.1)
                    attack_level = base_rate * attack_multiplier * intensity
                    noise = np.random.normal(0, attack_level * 0.15)
                    traffic[idx] = traffic[idx] + attack_level + noise
        
            # Phase 3: Ramp down (mitigation/filtering)
            for i in range(ramp_down_duration):
                idx = attack_start + ramp_up_duration + plateau_duration + i
                if idx < len(traffic):
                    progress = 1 - (i / ramp_down_duration)  # 1 to 0
                    intensity = progress ** 2
                    attack_level = base_rate * attack_multiplier * intensity
                    noise = np.random.normal(0, attack_level * 0.1)
                    traffic[idx] = traffic[idx] + attack_level + noise
    
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
        Generate training dataset - each scenario gets its own continuous block
         BUT we generate MULTIPLE instances of each scenario to increase diversity
        """
        all_data = []
    
        # Generate MULTIPLE instances of each scenario
        num_instances = 3  # Generate 3 different versions of each scenario
    
        scenarios_generators = {
            'normal': self.generate_normal_traffic,
            'burst': self.generate_burst_traffic,
            'congestion': self.generate_congestion_traffic,
            'ddos': self.generate_ddos_traffic,
            'mixed': self.generate_mixed_traffic
        }
    
        for instance in range(num_instances):
            print(f"  Generating instance {instance + 1}/{num_instances}...")
        
            for scenario_name, generator_func in scenarios_generators.items():
                # Generate new random instance
                traffic_data = generator_func()
            
                for t, bytes_val in zip(self.timestamps, traffic_data):
                    all_data.append({
                    'timestamp': t + (instance * self.duration),  # Offset time
                    'bytes_sent': int(bytes_val),
                    'scenario': scenario_name,
                    'instance': instance
                })
    
        df = pd.DataFrame(all_data)
    
        # IMPORTANT: Shuffle within each scenario to break time dependencies
        # But keep scenarios separate
        dfs_by_scenario = []
        for scenario in df['scenario'].unique():
            scenario_df = df[df['scenario'] == scenario].copy()
            # Don't shuffle - keep time series order
            dfs_by_scenario.append(scenario_df)
    
        # Concatenate all scenarios
        df = pd.concat(dfs_by_scenario, ignore_index=True)
    
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Generated {len(all_data)} samples across {len(scenarios_generators)} scenarios x {num_instances} instances")
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

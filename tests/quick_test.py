# quick_test.py
"""
Quick test with tiny dataset (1 minute instead of 10)
"""

import sys
sys.path.insert(0, 'src')

from ml_models.training.scenario_generator import TrafficScenarioGenerator

print("Quick Test: Generating small dataset...")

# Generate just 60 seconds per scenario
generator = TrafficScenarioGenerator(duration_seconds=60, sampling_rate=1.0)
df = generator.generate_all_scenarios('data/training/test_data.csv')

print(f"\n✓ Generated {len(df)} samples")
print("\nSample data:")
print(df.head(10))

print("\n✓ Quick test passed!")
print("Now run: python test_ml_pipeline.py")

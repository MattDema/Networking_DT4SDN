# test_ml_ultimate.py
"""
Ultimate optimized training with all fixes
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')

from ml_models.training.scenario_generator import TrafficScenarioGenerator
from ml_models.model_trainer import TrafficModelTrainer
from ml_models.traffic_predictor import TrafficPredictor

print("=" * 70)
print("ULTIMATE ML TRAINING - ALL OPTIMIZATIONS")
print("=" * 70)

# Generate 6 hours x 3 instances per scenario
generator = TrafficScenarioGenerator(duration_seconds=21600, sampling_rate=1.0)

os.makedirs('data/training/scenarios', exist_ok=True)

print("\n[STEP 1] Generating realistic training data...")
print("-" * 70)

scenarios_config = [
    ('normal', generator.generate_normal_traffic),
    ('burst', generator.generate_burst_traffic),
    ('congestion', generator.generate_congestion_traffic),
    ('ddos', generator.generate_ddos_traffic),
    ('mixed', generator.generate_mixed_traffic)
]

for scenario_name, generator_func in scenarios_config:
    print(f"\nGenerating {scenario_name}...")
    combined_traffic = []
    
    for instance in range(3):
        print(f"  Instance {instance+1}/3...")
        traffic = generator_func()
        combined_traffic.extend(traffic)
    
    # Convert list to numpy array
    combined_traffic = np.array(combined_traffic)

    timestamps = np.arange(len(combined_traffic))
    df = pd.DataFrame({
        'timestamp': timestamps,
        'bytes_sent': np.array(combined_traffic).astype(int),
        'scenario': scenario_name
    })
    
    df.to_csv(f'data/training/scenarios/{scenario_name}.csv', index=False)
    print(f"  ✓ {len(combined_traffic):,} samples")
    print(f"    Range: [{combined_traffic.min():.0f}, {combined_traffic.max():.0f}] bytes")
    print(f"    Mean: {np.mean(combined_traffic):.0f} bytes")

# Configuration per scenario
lr_config = {
    'normal': 0.0002,      # Faster learning for simple patterns
    'burst': 0.0001,
    'congestion': 0.00005, # Already working well, keep it
    'ddos': 0.0001,
    'mixed': 0.0001
}

model_config = {
    'normal': [128, 64, 32],
    'burst': [128, 64, 32],
    'congestion': [128, 64, 32],  # Already 0.8+, don't change
    'ddos': [256, 128, 64],       # Bigger model for complex patterns
    'mixed': [256, 128, 64]       # Bigger model for mixed patterns
}

sequence_length_config = {
    'normal': 60,       # Longer to capture daily patterns
    'burst': 90,
    'congestion': 90,   # Already working
    'ddos': 120,        # Longer to see attack buildup
    'mixed': 120
}

print("\n\n[STEP 2] Training with scenario-specific optimizations...")
print("-" * 70)

scenario_metrics = {}

for scenario_name, _ in scenarios_config:
    print(f"\n{'='*70}")
    print(f"Training {scenario_name.upper()}")
    print(f"  Sequence: {sequence_length_config[scenario_name]}s")
    print(f"  Architecture: {model_config[scenario_name]}")
    print(f"  Learning rate: {lr_config[scenario_name]}")
    print('='*70)
    
    trainer = TrafficModelTrainer(
        sequence_length=sequence_length_config[scenario_name],
        prediction_horizon=30,
        model_type='lstm'
    )
    
    trainer.load_data(f'data/training/scenarios/{scenario_name}.csv')
    trainer.build_model(units=model_config[scenario_name], dropout=0.3)
    
    trainer.train(
        epochs=1000,
        batch_size=128,
        learning_rate=lr_config[scenario_name],
        validation_split=0.15
    )
    
    metrics = trainer.evaluate()
    scenario_metrics[scenario_name] = metrics
    
    trainer.save_model(
        f'models/{scenario_name}_ultimate.pt',
        f'models/{scenario_name}_ultimate_scaler.pkl'
    )
    
    print(f"\n✓ {scenario_name} R²: {metrics['R2']:.4f}")
    
    if metrics['R2'] < 0.5:
        print(f"  !!! Low R² - This scenario needs more work")

# Summary
print("\n\n" + "=" * 70)
print("ULTIMATE TRAINING RESULTS")
print("=" * 70)

print(f"\n{'Scenario':<15} {'R² Score':<12} {'MAE (bytes)':<15} {'Status':<20}")
print("-" * 70)

for scenario_name, metrics in scenario_metrics.items():
    r2 = metrics['R2']
    
    if r2 > 0.8:
        status = "✓✓ Excellent"
    elif r2 > 0.6:
        status = "✓ Good"
    elif r2 > 0.4:
        status = "+ Acceptable"
    else:
        status = "X Needs work"
    
    print(f"{scenario_name:<15} {r2:<12.4f} {metrics['MAE']:<15.1f} {status:<20}")

avg_r2 = np.mean([m['R2'] for m in scenario_metrics.values()])
print(f"\n{'AVERAGE':<15} {avg_r2:<12.4f}")

if avg_r2 > 0.7:
    print("\n Overall performance is EXCELLENT!")
    print("   Models are ready for production deployment!")
elif avg_r2 > 0.5:
    print("\n✓ Overall performance is GOOD")
    print("  Models can be used, but some scenarios might need tuning")
else:
    print(f"\n Average R² = {avg_r2:.4f}")
    print("  Consider:")
    print("  1. Check if scenario generation is realistic")
    print("  2. Increase training duration further")
    print("  3. Try GRU instead of LSTM")

print("\n" + "=" * 70)

# Detailed analysis
print("\n[BONUS] Per-Scenario Insights:")
print("-" * 70)

for scenario_name, metrics in scenario_metrics.items():
    print(f"\n{scenario_name.upper()}:")
    if metrics['R2'] > 0.7:
        print(f"  ✓ Excellent prediction capability")
        print(f"  ✓ Can accurately forecast traffic patterns")
    elif metrics['R2'] > 0.4:
        print(f"    Moderate prediction - usable but not optimal")
        print(f"    MAE: {metrics['MAE']:.1f} bytes average error")
    else:
        print(f"    Poor prediction - investigate data patterns")
        print(f"    Possible issues:")
        print(f"    - Data too random/noisy")
        print(f"    - Patterns too complex for current architecture")
        print(f"    - Need longer sequence or different model")

print("\n" + "=" * 70)

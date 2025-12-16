"""
Final optimized training with all fixes applied
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
print("FINAL OPTIMIZED ML TRAINING")
print("=" * 70)

# Generate 6 hours per scenario x 3 instances = 18 hours total per scenario
generator = TrafficScenarioGenerator(duration_seconds=21600, sampling_rate=1.0)

os.makedirs('data/training/scenarios', exist_ok=True)

print("\n[STEP 1] Generating diverse training data...")
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
    
    # Generate 3 different instances for diversity
    for instance in range(3):
        print(f"  Instance {instance+1}/3...")
        traffic = generator_func()
        combined_traffic.extend(traffic)
    
    # Save
    timestamps = np.arange(len(combined_traffic))
    df = pd.DataFrame({
        'timestamp': timestamps,
        'bytes_sent': np.array(combined_traffic).astype(int),
        'scenario': scenario_name
    })
    
    df.to_csv(f'data/training/scenarios/{scenario_name}.csv', index=False)
    print(f"  ✓ {len(combined_traffic):,} samples saved")

# Train models
print("\n\n[STEP 2] Training optimized models...")
print("-" * 70)

scenario_metrics = {}

# Model size per scenario
model_config = {
    'normal': [128, 64, 32],        # Standard
    'burst': [128, 64, 32],         # Standard
    'congestion': [128, 64, 32],    # Standard (working well)
    'ddos': [256, 128, 64],         # LARGER for complex patterns
    'mixed': [256, 128, 64]         # LARGER for mixed patterns
}

# Learning rate per scenario (some need faster learning)
lr_config = {
    'normal': 0.0002,      # Faster - simple patterns need less precision
    'burst': 0.0001,
    'congestion': 0.00005, # Slower - complex gradual patterns
    'ddos': 0.0001,
    'mixed': 0.0001
}

for scenario_name, _ in scenarios_config:
    print(f"\n{'='*70}")
    print(f"Training {scenario_name.upper()}")
    print('='*70)
    
    trainer = TrafficModelTrainer(
        sequence_length=120,
        prediction_horizon=30,
        model_type='lstm'
    )
    
    trainer.load_data(f'data/training/scenarios/{scenario_name}.csv')
    # trainer.build_model(units=[128, 64, 32], dropout=0.2)
    trainer.build_model(
        units=model_config[scenario_name],  # Scenario-specific architecture
        dropout=0.2
    )
    
    # OPTIMIZED PARAMETERS:
    # - Lower learning rate
    # - More epochs
    # - Larger batch
    # trainer.train(
    #     epochs=900,           # More epochs
    #     batch_size=128,       # Larger batch for stability
    #     learning_rate=0.0001, # Slower learning
    #     validation_split=0.15
    # )
    trainer.train(
        epochs=1000,            # More epochs
        batch_size=128,          # Larger batch for stability
        learning_rate=lr_config[scenario_name],  # Use scenario-specific LR
        validation_split=0.15
    )
    
    metrics = trainer.evaluate()
    scenario_metrics[scenario_name] = metrics
    
    trainer.save_model(
        f'models/{scenario_name}_final.pt',
        f'models/{scenario_name}_final_scaler.pkl'
    )
    
    print(f"\n✓ {scenario_name} R²: {metrics['R2']:.4f}")

# Summary
print("\n\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"\n{'Scenario':<15} {'R²':<10} {'MAE':<10} {'RMSE':<10} {'Status':<15}")
print("-" * 70)



for scenario_name, metrics in scenario_metrics.items():
    r2 = metrics['R2']
    status = "✓ Excellent" if r2 > 0.8 else "✓ Good" if r2 > 0.6 else "⚠️ Needs work"
    print(f"{scenario_name:<15} {r2:<10.4f} {metrics['MAE']:<10.1f} {metrics['RMSE']:<10.1f} {status:<15}")

avg_r2 = np.mean([m['R2'] for m in scenario_metrics.items()])
print(f"\n{'AVERAGE':<15} {avg_r2:<10.4f}")

if avg_r2 > 0.7:
    print("\n Models ready for production!")
else:
    print(f"\n Average R² = {avg_r2:.4f} - May need more tuning")

print("=" * 70)

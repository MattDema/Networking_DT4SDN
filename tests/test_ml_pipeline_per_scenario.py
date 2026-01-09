# test_ml_pipeline_per_scenario.py
"""
Train separate models for each scenario type
This gives much better R² scores
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
print("PER-SCENARIO ML TRAINING")
print("=" * 70)

# ============================================
# Generate Data for Each Scenario
# ============================================
print("\n[STEP 1] Generating training data per scenario...")
print("-" * 70)

generator = TrafficScenarioGenerator(duration_seconds=21600, sampling_rate=1.0)  # 6 hours

scenarios = {
    'normal': generator.generate_normal_traffic(),
    'burst': generator.generate_burst_traffic(),
    'congestion': generator.generate_congestion_traffic(),
    'ddos': generator.generate_ddos_traffic(),
    'mixed': generator.generate_mixed_traffic()
}

print("\nGenerating multiple instances for robustness...")

all_scenarios_data = {}

for scenario_name, generator_func in [
    ('normal', generator.generate_normal_traffic),
    ('burst', generator.generate_burst_traffic),
    ('congestion', generator.generate_congestion_traffic),
    ('ddos', generator.generate_ddos_traffic),
    ('mixed', generator.generate_mixed_traffic)
]:

# Generate 3 different instances of each scenario
    combined_traffic = []
    
    for instance in range(3):
        print(f"  Generating {scenario_name} instance {instance+1}/3...")
        traffic = generator_func()
        combined_traffic.extend(traffic)
    
    # Now we have 3x the data
    timestamps = np.arange(len(combined_traffic))
    df = pd.DataFrame({
        'timestamp': timestamps,
        'bytes_sent': np.array(combined_traffic).astype(int),
        'scenario': scenario_name
    })
    
    df.to_csv(f'data/training/scenarios/{scenario_name}.csv', index=False)
    print(f"  ✓ {scenario_name}: {len(combined_traffic)} samples (3 instances)")

# Save each scenario separately
os.makedirs('data/training/scenarios', exist_ok=True)

for scenario_name, traffic_data in scenarios.items():
    df = pd.DataFrame({
        'timestamp': generator.timestamps,
        'bytes_sent': traffic_data.astype(int),
        'scenario': scenario_name
    })
    df.to_csv(f'data/training/scenarios/{scenario_name}.csv', index=False)
    print(f"  ✓ {scenario_name}: {len(traffic_data)} samples, "
          f"range [{traffic_data.min():.0f}, {traffic_data.max():.0f}] bytes")

# ============================================
# Train Model for Each Scenario
# ============================================
print("\n\n[STEP 2] Training models for each scenario...")
print("-" * 70)

scenario_models = {}
scenario_metrics = {}

for scenario_name in scenarios.keys():
    print(f"\n{'='*70}")
    print(f"Training {scenario_name.upper()} model")
    print('='*70)
    
    trainer = TrafficModelTrainer(
        sequence_length=60,
        prediction_horizon=30,
        model_type='lstm'
    )
    
    # Load scenario-specific data
    trainer.load_data(f'data/training/scenarios/{scenario_name}.csv')
    
    # Build model
    trainer.build_model(units=[128, 64, 32], dropout=0.2)
    
    # Train
    trainer.train(
        epochs=1000,
        batch_size=64,
        learning_rate=0.00001,
        validation_split=0.15
    )
    
    # Evaluate
    metrics = trainer.evaluate()
    scenario_metrics[scenario_name] = metrics
    
    # Save model
    model_path = f'models/{scenario_name}_model.pt'
    scaler_path = f'models/{scenario_name}_scaler.pkl'
    trainer.save_model(model_path, scaler_path)
    
    scenario_models[scenario_name] = {
        'model_path': model_path,
        'scaler_path': scaler_path,
        'metrics': metrics
    }
    
    print(f"\n✓ {scenario_name} R²: {metrics['R2']:.4f}")

# ============================================
# Summary
# ============================================
print("\n\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

print("\nPer-Scenario Performance:")
print(f"{'Scenario':<15} {'R² Score':<12} {'MAE':<12} {'RMSE':<12}")
print("-" * 70)

for scenario_name, metrics in scenario_metrics.items():
    print(f"{scenario_name:<15} {metrics['R2']:<12.4f} {metrics['MAE']:<12.2f} {metrics['RMSE']:<12.2f}")

avg_r2 = np.mean([m['R2'] for m in scenario_metrics.values()])
print(f"\n{'AVERAGE':<15} {avg_r2:<12.4f}")

if avg_r2 > 0.8:
    print("\n🎉 Excellent performance across all scenarios!")
elif avg_r2 > 0.6:
    print("\n✓ Good performance - ready for production")
else:
    print("\n⚠️  Consider training longer or adjusting hyperparameters")

print("\n" + "=" * 70)

# ============================================
# Test Predictions
# ============================================
print("\n[STEP 3] Testing predictions...")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, (scenario_name, scenario_data) in enumerate(scenarios.items()):
    print(f"\nTesting {scenario_name}...")
    
    # Load scenario-specific model
    predictor = TrafficPredictor(
        scenario_models[scenario_name]['model_path'],
        scenario_models[scenario_name]['scaler_path']
    )
    
    # Test on middle section
    start_idx = len(scenario_data) // 2
    input_sequence = scenario_data[start_idx:start_idx + 60]
    actual_future = scenario_data[start_idx + 60:start_idx + 90]
    
    # Predict
    predicted_future = predictor.predict(input_sequence)
    
    # Calculate error
    mae = np.mean(np.abs(predicted_future - actual_future))
    r2 = scenario_metrics[scenario_name]['R2']
    
    # Plot
    time_axis = np.arange(90)
    axes[idx].plot(time_axis[:60], input_sequence, 'b-', label='Historical', linewidth=2.5, alpha=0.8)
    axes[idx].plot(time_axis[60:], actual_future, 'g-', label='Actual', linewidth=2.5)
    axes[idx].plot(time_axis[60:], predicted_future, 'r--', label='Predicted', linewidth=2.5)
    axes[idx].axvline(x=60, color='black', linestyle=':', alpha=0.5, linewidth=2)
    axes[idx].fill_between(time_axis[60:], actual_future, predicted_future, alpha=0.2, color='red')
    
    axes[idx].set_title(f'{scenario_name.upper()}\nR²: {r2:.3f} | MAE: {mae:.1f} bytes',
                       fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Time (seconds)', fontsize=11)
    axes[idx].set_ylabel('Traffic (bytes/s)', fontsize=11)
    axes[idx].legend(loc='upper left', fontsize=10)
    axes[idx].grid(True, alpha=0.3)

fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig('docs/per_scenario_predictions.png', dpi=150)
print("\n✓ Predictions saved to: docs/per_scenario_predictions.png")

print("\n" + "=" * 70)
print("✓ ALL MODELS TRAINED AND TESTED!")
print("=" * 70)

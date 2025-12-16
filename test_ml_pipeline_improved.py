# test_ml_pipeline_improved.py
"""
Improved ML pipeline with better hyperparameters
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')

from ml_models.training.scenario_generator import TrafficScenarioGenerator
from ml_models.model_trainer import TrafficModelTrainer
from ml_models.traffic_predictor import TrafficPredictor

print("=" * 70)
print("IMPROVED ML PREDICTION PIPELINE")
print("=" * 70)

# ============================================
# STEP 1: Generate More Training Data
# ============================================
print("\n[STEP 1/5] Generating training data...")
print("-" * 70)

# Generate MORE data - 2 hours per scenario instead of 1
generator = TrafficScenarioGenerator(duration_seconds=7200, sampling_rate=1.0)
df = generator.generate_all_scenarios('data/training/traffic_scenarios_improved.csv')

print(f"\n✓ Generated {len(df)} training samples")
print(f"  Scenarios: {df['scenario'].unique()}")

# ============================================
# STEP 2: Train with Better Architecture
# ============================================
print("\n\n[STEP 2/5] Training ML model with improved settings...")
print("-" * 70)

trainer = TrafficModelTrainer(
    sequence_length=60,
    prediction_horizon=30,
    model_type='lstm'
)

trainer.load_data('data/training/traffic_scenarios_improved.csv')

# IMPROVED: Deeper network with more capacity
trainer.build_model(units=[128, 64, 32], dropout=0.3)

# IMPROVED: Lower learning rate, larger batch, more epochs
print("\nTraining for 150 epochs with improved settings...")
trainer.train(
    epochs=150,
    batch_size=64,
    learning_rate=0.0001,  # 10x slower learning
    validation_split=0.15  # More validation data
)

# ============================================
# STEP 3: Evaluate
# ============================================
print("\n\n[STEP 3/5] Evaluating model...")
print("-" * 70)

metrics = trainer.evaluate()

print("\n✓ Model Performance:")
print(f"  MAE (Mean Absolute Error):  {metrics['MAE']:.2f} bytes")
print(f"  RMSE (Root Mean Sq Error):  {metrics['RMSE']:.2f} bytes")
print(f"  R² Score:                   {metrics['R2']:.4f}")

if metrics['R2'] > 0.8:
    print("  → Excellent prediction accuracy! ✓")
elif metrics['R2'] > 0.6:
    print("  → Good prediction accuracy")
elif metrics['R2'] > 0.4:
    print("  → Moderate accuracy, acceptable")
else:
    print("  → Needs more tuning")

# ============================================
# STEP 4: Save Model
# ============================================
print("\n\n[STEP 4/5] Saving model...")
print("-" * 70)

os.makedirs('models', exist_ok=True)
trainer.save_model('models/traffic_model_improved.pt', 'models/scaler_improved.pkl')
trainer.plot_training_history('docs/training_history_improved.png')

# ============================================
# STEP 5: Test Predictions with Visualization
# ============================================
print("\n\n[STEP 5/5] Testing predictions...")
print("-" * 70)

predictor = TrafficPredictor('models/traffic_model_improved.pt', 'models/scaler_improved.pkl')

print("\n✓ Model loaded:")
info = predictor.get_model_info()
for key, value in info.items():
    print(f"  {key}: {value}")

# Test predictions
print("\n\nTesting predictions on each scenario:")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

scenario_errors = {}

for idx, scenario in enumerate(['normal', 'burst', 'congestion', 'ddos', 'mixed']):
    scenario_data = df[df['scenario'] == scenario]['bytes_sent'].values
    
    # Use middle section for testing (avoid edges)
    start_idx = len(scenario_data) // 3
    input_sequence = scenario_data[start_idx:start_idx + 60]
    actual_future = scenario_data[start_idx + 60:start_idx + 90]
    
    # Predict
    predicted_future = predictor.predict(input_sequence)
    
    # Calculate errors
    mae = np.mean(np.abs(predicted_future - actual_future))
    mape = np.mean(np.abs((predicted_future - actual_future) / (actual_future + 1))) * 100
    scenario_errors[scenario] = {'MAE': mae, 'MAPE': mape}
    
    # Plot
    time_axis = np.arange(90)
    axes[idx].plot(time_axis[:60], input_sequence, 'b-', label='Historical', linewidth=2)
    axes[idx].plot(time_axis[60:], actual_future, 'g-', label='Actual', linewidth=2)
    axes[idx].plot(time_axis[60:], predicted_future, 'r--', label='Predicted', linewidth=2, alpha=0.8)
    axes[idx].axvline(x=60, color='black', linestyle=':', alpha=0.5, linewidth=1.5)
    axes[idx].fill_between(time_axis[60:], actual_future, predicted_future, alpha=0.2, color='red')
    
    axes[idx].set_title(f'{scenario.upper()}\nMAE: {mae:.1f} bytes | MAPE: {mape:.1f}%', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Time (seconds)')
    axes[idx].set_ylabel('Traffic (bytes/s)')
    axes[idx].legend(loc='upper left')
    axes[idx].grid(True, alpha=0.3)

fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig('docs/prediction_examples_improved.png', dpi=150)
print("\n✓ Prediction examples saved to: docs/prediction_examples_improved.png")

# Print summary
print("\n\nPer-Scenario Performance:")
print("-" * 70)
for scenario, errors in scenario_errors.items():
    print(f"{scenario.upper():12s} - MAE: {errors['MAE']:6.1f} bytes | MAPE: {errors['MAPE']:5.1f}%")

# ============================================
# Summary
# ============================================
print("\n\n" + "=" * 70)
print("IMPROVED TRAINING COMPLETE!")
print("=" * 70)

print("\n✓ Model saved to: models/traffic_model_improved.pt")
print(f"✓ Overall R² Score: {metrics['R2']:.4f}")
print("\n✓ Ready for integration with RYU controller!")

if metrics['R2'] > 0.7:
    print("\n Model performance is good enough for production use!")
else:
    print(f"\n R² = {metrics['R2']:.4f} - Consider training longer or collecting more data")

print("=" * 70)

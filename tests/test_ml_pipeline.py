# test_ml_pipeline.py
"""
End-to-end test of ML prediction pipeline
Tests: scenario generation → training → prediction → visualization
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, 'src')

from ml_models.training.scenario_generator import TrafficScenarioGenerator
from ml_models.model_trainer import TrafficModelTrainer
from ml_models.traffic_predictor import TrafficPredictor

print("=" * 70)
print("ML PREDICTION PIPELINE TEST")
print("=" * 70)

# ============================================
# STEP 1: Generate Training Data
# ============================================
print("\n[STEP 1/5] Generating training data...")
print("-" * 70)

generator = TrafficScenarioGenerator(duration_seconds=3600, sampling_rate=1.0) # 1 hour per scenario
df = generator.generate_all_scenarios('data/training/traffic_scenarios.csv')

print(f"\n✓ Generated {len(df)} training samples")
print(f"  Scenarios: {df['scenario'].unique()}")
print(f"\nTraffic statistics by scenario:")
print(df.groupby('scenario')['bytes_sent'].describe()[['mean', 'std', 'min', 'max']])

# ============================================
# STEP 2: Train Model
# ============================================
print("\n\n[STEP 2/5] Training ML model...")
print("-" * 70)

trainer = TrafficModelTrainer(
    sequence_length=60,      # Look back 60 seconds
    prediction_horizon=30,   # Predict next 30 seconds
    model_type='lstm'
)

trainer.load_data('data/training/traffic_scenarios.csv')
trainer.build_model(units=[128, 64, 32], dropout=0.2)

# Train (quick test with fewer epochs)
print("\nTraining for 20 epochs (use 50-100 for production)...")
trainer.train(epochs=100, batch_size=64, learning_rate=0.0001)

# ============================================
# STEP 3: Evaluate Model
# ============================================
print("\n\n[STEP 3/5] Evaluating model...")
print("-" * 70)

metrics = trainer.evaluate()

print("\n✓ Model Performance:")
print(f"  MAE (Mean Absolute Error):  {metrics['MAE']:.2f} bytes")
print(f"  RMSE (Root Mean Sq Error):  {metrics['RMSE']:.2f} bytes")
print(f"  R² Score:                   {metrics['R2']:.4f}")

if metrics['R2'] > 0.7:
    print("  → Good prediction accuracy!")
elif metrics['R2'] > 0.5:
    print("  → Moderate accuracy, consider more training")
else:
    print("  → Low accuracy, needs more data or tuning")

# ============================================
# STEP 4: Save Model
# ============================================
print("\n\n[STEP 4/5] Saving model...")
print("-" * 70)

os.makedirs('models', exist_ok=True)
trainer.save_model('models/traffic_model.pt', 'models/scaler.pkl')
trainer.plot_training_history('docs/training_history.png')

# ============================================
# STEP 5: Test Prediction
# ============================================
print("\n\n[STEP 5/5] Testing predictions...")
print("-" * 70)

# Load trained model
predictor = TrafficPredictor('models/traffic_model.pt', 'models/scaler.pkl')

print("\n✓ Model loaded:")
info = predictor.get_model_info()
for key, value in info.items():
    print(f"  {key}: {value}")

# Test on different scenarios
print("\n\nTesting predictions on each scenario:")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, scenario in enumerate(['normal', 'burst', 'congestion', 'ddos', 'mixed']):
    # Get sample data for this scenario
    scenario_data = df[df['scenario'] == scenario]['bytes_sent'].values
    
    # Use first 60 seconds as input
    input_sequence = scenario_data[:60]
    
    # Ground truth: actual next 30 seconds
    actual_future = scenario_data[60:90]
    
    # Predict
    predicted_future = predictor.predict(input_sequence)
    
    # Calculate error
    mae = np.mean(np.abs(predicted_future - actual_future))
    
    # Plot
    axes[idx].plot(range(60), input_sequence, 'b-', label='Historical (input)', linewidth=2)
    axes[idx].plot(range(60, 90), actual_future, 'g-', label='Actual future', linewidth=2)
    axes[idx].plot(range(60, 90), predicted_future, 'r--', label='Predicted', linewidth=2)
    axes[idx].axvline(x=60, color='black', linestyle=':', alpha=0.5)
    axes[idx].set_title(f'{scenario.upper()}\nMAE: {mae:.1f} bytes')
    axes[idx].set_xlabel('Time (seconds)')
    axes[idx].set_ylabel('Traffic (bytes/s)')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

# Remove empty subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('docs/prediction_examples.png', dpi=150)
print("\n✓ Prediction examples saved to: docs/prediction_examples.png")

# ============================================
# Summary
# ============================================
print("\n\n" + "=" * 70)
print("TEST COMPLETE!")
print("=" * 70)

print("\n✓ Files created:")
print("  data/training/traffic_scenarios.csv  - Training data")
print("  models/traffic_model.pt              - Trained model")
print("  models/scaler.pkl                    - Data normalizer")
print("  docs/training_history.png            - Training curves")
print("  docs/prediction_examples.png         - Prediction results")
print("  docs/traffic_scenarios.png           - Scenario patterns")

print("\n✓ Model is ready to use!")
print("\nNext steps:")
print("  1. Integrate with RYU controller (src/controllers/)")
print("  2. Build web interface (src/web_interface/)")
print("  3. Test on real Mininet topology")

print("\n" + "=" * 70)

""" # src/ml_models/training/train_with_scenarios.py

from scenario_generator import TrafficScenarioGenerator
from model_trainer import TrafficModelTrainer

# Step 1: Generate training data
print("Generating traffic scenarios...")
generator = TrafficScenarioGenerator(duration_seconds=3600)  # 1 hour each
df = generator.generate_all_scenarios('data/training/traffic_scenarios.csv')

# Step 2: Train model
print("\nTraining ML model...")
trainer = TrafficModelTrainer(
    sequence_length=60,
    prediction_horizon=30,
    model_type='lstm'
)

trainer.load_data('data/training/traffic_scenarios.csv')
trainer.build_model(units=[128, 64], dropout=0.2)
trainer.train(epochs=100, batch_size=64)

# Step 3: Evaluate
metrics = trainer.evaluate()
trainer.save_model('models/traffic_model_scenarios.pt')

print("\n✓ Model trained on all scenarios!")
print(f"  Can now predict: Normal, Burst, Congestion, DDoS, Mixed patterns")
 """
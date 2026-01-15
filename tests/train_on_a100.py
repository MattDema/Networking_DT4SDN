# train_on_a100.py
"""
ULTIMATE A100-40GB Training Script
Pushes hardware to maximum potential
"""

import torch
import sys
import os
import time
sys.path.insert(0, 'src')

from ml_models.model_trainer import TrafficModelTrainer
import numpy as np
from datetime import datetime

print("="*80)
print("🚀 TRAINING ON NVIDIA A100-40GB - MAXIMUM POWER MODE 🚀")
print("="*80)

# Verify GPU
if not torch.cuda.is_available():
    print("\n❌ ERROR: No GPU detected! Make sure you selected A100 in Colab.")
    print("   Runtime → Change runtime type → Hardware accelerator → A100 GPU")
    exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"\n✓ GPU: {gpu_name}")
print(f"✓ Memory: {gpu_memory:.1f} GB")
print(f"✓ CUDA Version: {torch.version.cuda}")

if "A100" not in gpu_name:
    print("\n⚠️  WARNING: Not using A100! You selected a different GPU.")
    proceed = input("Continue anyway? [y/N]: ")
    if proceed.lower() != 'y':
        exit(1)

# EXTREME A100-40GB Configuration
A100_CONFIG = {
    'sequence_length': 240,
    'prediction_horizon': 90,
    'batch_size': 1024,
    'epochs': 5000,
    'learning_rate': 0.0003,
    'patience': 500,
    'units': [512, 512, 256, 128, 64],
    'dropout': 0.3,
    'use_mixed_precision': True,
    'validation_split': 0.15
}

print("\n" + "="*80)
print("⚡ EXTREME A100 CONFIGURATION ⚡")
print("="*80)
print(f"  Sequence Length:      {A100_CONFIG['sequence_length']}s (3x longer!)")
print(f"  Prediction Horizon:   {A100_CONFIG['prediction_horizon']}s (3x further!)")
print(f"  Batch Size:           {A100_CONFIG['batch_size']} (16x larger!)")
print(f"  Network Architecture: {A100_CONFIG['units']}")
print(f"  Network Depth:        {len(A100_CONFIG['units'])} layers (DEEP!)")
print(f"  Max Epochs:           {A100_CONFIG['epochs']}")
print(f"  Patience:             {A100_CONFIG['patience']}")
print(f"  Mixed Precision:      {A100_CONFIG['use_mixed_precision']} (CRITICAL!)")
print("="*80)

# Scenarios to train
scenarios = ['normal', 'burst', 'congestion', 'ddos', 'mixed']

results = {}
total_start = time.time()

for idx, scenario in enumerate(scenarios, 1):
    print(f"\n\n{'='*80}")
    print(f"[{idx}/{len(scenarios)}] TRAINING: {scenario.upper()}")
    print('='*80)
    
    scenario_start = time.time()
    
    try:
        # Initialize trainer
        trainer = TrafficModelTrainer(
            sequence_length=A100_CONFIG['sequence_length'],
            prediction_horizon=A100_CONFIG['prediction_horizon'],
            model_type='lstm'
        )
        
        # Load data
        data_path = f'data/training/scenarios/{scenario}.csv'
        if not os.path.exists(data_path):
            print(f"⚠️  WARNING: Data not found at {data_path}")
            print(f"   Skipping {scenario}...")
            continue
        
        print(f"\n[1/5] Loading data from {data_path}...")
        trainer.load_data(data_path)
        
        # Build DEEP model
        print(f"\n[2/5] Building DEEP {len(A100_CONFIG['units'])}-layer LSTM model...")
        trainer.build_model(
            units=A100_CONFIG['units'],
            dropout=A100_CONFIG['dropout']
        )
        
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"  ✓ Total parameters: {total_params:,}")
        print(f"  ✓ Trainable parameters: {trainable_params:,}")
        print(f"  ✓ Estimated model size: {total_params * 4 / 1e6:.1f} MB")
        
        print(f"\n[3/5] Starting training with A100 EXTREME config...")
        print(f"      This may take 20-45 minutes depending on early stopping...")
        
        training_start = time.time()
        
        # TRAIN with extreme config
        trainer.train(
            epochs=A100_CONFIG['epochs'],
            batch_size=A100_CONFIG['batch_size'],
            learning_rate=A100_CONFIG['learning_rate'],
            validation_split=A100_CONFIG['validation_split'],
            use_mixed_precision=A100_CONFIG['use_mixed_precision']
        )
        
        training_time = time.time() - training_start
        
        print(f"\n[4/5] Evaluating model on test set...")
        metrics = trainer.evaluate()
        
        # Calculate throughput
        total_samples = len(trainer.X_train) + len(trainer.X_test)
        samples_per_second = total_samples * len(trainer.train_losses) / training_time
        
        results[scenario] = {
            'R2': metrics['R2'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'training_time_min': training_time / 60,
            'total_params': total_params,
            'epochs_trained': len(trainer.train_losses),
            'samples_per_sec': samples_per_second
        }
        
        print(f"\n{'='*80}")
        print(f"📊 RESULTS: {scenario.upper()}")
        print('='*80)
        print(f"  R² Score:          {metrics['R2']:.4f}")
        print(f"  MAE:               {metrics['MAE']:.2f} bytes")
        print(f"  RMSE:              {metrics['RMSE']:.2f} bytes")
        print(f"  Training Time:     {training_time/60:.1f} minutes")
        print(f"  Epochs Trained:    {len(trainer.train_losses)}")
        print(f"  Throughput:        {samples_per_second:.0f} samples/sec")
        
        if metrics['R2'] > 0.9:
            status = "🎉 EXCELLENT!"
            emoji = "🏆"
        elif metrics['R2'] > 0.8:
            status = "🎉 VERY GOOD!"
            emoji = "✨"
        elif metrics['R2'] > 0.7:
            status = "✓ GOOD"
            emoji = "✓"
        elif metrics['R2'] > 0.5:
            status = "✓ Acceptable"
            emoji = "⚠️"
        else:
            status = "❌ POOR - needs work"
            emoji = "❌"
        
        print(f"  Status:            {emoji} {status}")
        print('='*80)
        
        # Save model
        print(f"\n[5/5] Saving model...")
        model_path = f'models/{scenario}_a100_extreme.pt'
        scaler_path = f'models/{scenario}_a100_extreme_scaler.pkl'
        trainer.save_model(model_path, scaler_path)
        print(f"  ✓ Model saved: {model_path}")
        
        scenario_time = time.time() - scenario_start
        print(f"\n✓ {scenario.upper()} complete in {scenario_time/60:.1f} minutes")
        
    except Exception as e:
        print(f"\n❌ ERROR training {scenario}:")
        import traceback
        traceback.print_exc()
        continue

# Final Summary
total_time = time.time() - total_start

print("\n\n" + "="*80)
print("🏁 TRAINING COMPLETE - FINAL SUMMARY 🏁")
print("="*80)

print(f"\n{'Scenario':<12} {'R²':<8} {'MAE':<10} {'Time(min)':<10} {'Epochs':<8} {'Status'}")
print("-"*80)

for scenario, m in results.items():
    status = "🏆" if m['R2'] > 0.9 else "✨" if m['R2'] > 0.8 else "✓" if m['R2'] > 0.7 else "⚠️" if m['R2'] > 0.5 else "❌"
    print(f"{scenario:<12} {m['R2']:<8.4f} {m['MAE']:<10.2f} {m['training_time_min']:<10.1f} "
          f"{m['epochs_trained']:<8} {status}")

print("-"*80)
print(f"{'TOTAL TIME:':<12} {'':<8} {'':<10} {total_time/60:<10.1f} minutes")
print("="*80)

# Performance stats
avg_r2 = np.mean([m['R2'] for m in results.values()])
print(f"\n📊 Average R² Score: {avg_r2:.4f}")
print(f"⏱️  Total Training Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"✓ Models saved to: models/*_a100_extreme.pt")

excellent_count = sum(1 for m in results.values() if m['R2'] > 0.9)
good_count = sum(1 for m in results.values() if 0.7 <= m['R2'] <= 0.9)
poor_count = sum(1 for m in results.values() if m['R2'] < 0.5)

print(f"\n🏆 Excellent models (R² > 0.9): {excellent_count}/{len(results)}")
print(f"✨ Good models (R² > 0.7):      {good_count}/{len(results)}")
print(f"❌ Poor models (R² < 0.5):      {poor_count}/{len(results)}")

print("\n" + "="*80)
print("🚀 A100 EXTREME TRAINING COMPLETE! 🚀")
print("="*80)

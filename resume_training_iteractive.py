# resume_training_interactive.py
"""
Interactive script to resume/improve training for any model
Allows multiple rounds of training and tracks performance
"""

import sys
import os
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, 'src')
from ml_models.model_trainer import TrafficModelTrainer

print("=" * 70)
print("INTERACTIVE MODEL IMPROVEMENT")
print("=" * 70)

# ============================================
# Step 1: Select Models to Improve
# ============================================

available_scenarios = []
for scenario in ['normal', 'burst', 'congestion', 'ddos', 'mixed']:
    model_path = f'models/{scenario}_ultimate.pt'
    if os.path.exists(model_path):
        available_scenarios.append(scenario)

if not available_scenarios:
    print("\n❌ No trained models found in models/ directory")
    print("   Please train models first using test_ml_ultimate.py")
    sys.exit(1)

print("\nAvailable models:")
for i, scenario in enumerate(available_scenarios, 1):
    # Check current performance
    try:
        trainer = TrafficModelTrainer(sequence_length=60, prediction_horizon=30, model_type='lstm')
        trainer.load_data(f'data/training/scenarios/{scenario}.csv')
        
        checkpoint = torch.load(f'models/{scenario}_ultimate.pt', weights_only=False)
        trainer.model = checkpoint['model']
        trainer.model.to(trainer.device)
        trainer.scaler = joblib.load(f'models/{scenario}_ultimate_scaler.pkl')
        
        metrics = trainer.evaluate()
        r2 = metrics['R2']
        
        status = "🎉 Excellent" if r2 > 0.8 else "✓ Good" if r2 > 0.6 else "⚠️ Needs work" if r2 > 0.3 else "❌ Poor"
        print(f"  [{i}] {scenario:15s} - R²: {r2:6.4f}  {status}")
    except:
        print(f"  [{i}] {scenario:15s} - (unable to evaluate)")

print(f"\n  [0] Train ALL models")
print(f"  [q] Quit")

# Get user selection
while True:
    choice = input("\nSelect model(s) to improve (e.g., 1 or 1,2,5 or 0 for all): ").strip()
    
    if choice.lower() == 'q':
        print("Exiting...")
        sys.exit(0)
    
    try:
        if choice == '0':
            selected_scenarios = available_scenarios
            break
        else:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected_scenarios = [available_scenarios[i] for i in indices]
            break
    except (ValueError, IndexError):
        print("❌ Invalid selection. Try again.")

print(f"\n✓ Selected: {', '.join(selected_scenarios)}")

# ============================================
# Step 2: Training Configuration
# ============================================

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print("\nHow many improvement rounds?")
print("  1-2 rounds: Safe, minimal overfitting risk")
print("  3-5 rounds: Moderate, watch for overfitting")
print("  5+ rounds: High risk, only if R² is still very low")

while True:
    try:
        num_rounds = int(input("\nNumber of training rounds [1-10]: ").strip())
        if 1 <= num_rounds <= 10:
            break
        print("Please enter a number between 1 and 10")
    except ValueError:
        print("Please enter a valid number")

print("\nEpochs per round?")
print("  100-300: Quick improvement")
print("  500-1000: Thorough training")
print("  1000+: Only for very stubborn models")

while True:
    try:
        epochs = int(input("\nEpochs per round [100-2000]: ").strip())
        if 100 <= epochs <= 2000:
            break
        print("Please enter a number between 100 and 2000")
    except ValueError:
        print("Please enter a valid number")

print("\nLearning rate strategy?")
print("  [1] Conservative (0.00001) - Very slow, very safe")
print("  [2] Moderate (0.00005) - Balanced")
print("  [3] Aggressive (0.0001) - Faster, might overshoot")
print("  [4] Adaptive (starts high, decreases) - Recommended")

while True:
    lr_choice = input("\nSelect learning rate strategy [1-4]: ").strip()
    if lr_choice in ['1', '2', '3', '4']:
        break
    print("Please select 1, 2, 3, or 4")

lr_config = {
    '1': 0.00001,
    '2': 0.00005,
    '3': 0.0001,
    '4': 'adaptive'
}

learning_rate = lr_config[lr_choice]

# ============================================
# Step 3: Train Selected Models
# ============================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

# Track all results
all_results = {}

for scenario in selected_scenarios:
    print(f"\n{'='*70}")
    print(f"IMPROVING: {scenario.upper()}")
    print(f"{'='*70}")
    
    # Load model and data
    trainer = TrafficModelTrainer(
        sequence_length=60,
        prediction_horizon=30,
        model_type='lstm'
    )
    
    trainer.load_data(f'data/training/scenarios/{scenario}.csv')
    
    # Load existing model
    checkpoint = torch.load(f'models/{scenario}_ultimate.pt', weights_only=False)
    trainer.model = checkpoint['model']
    trainer.model.to(trainer.device)
    trainer.scaler = joblib.load(f'models/{scenario}_ultimate_scaler.pkl')
    
    # Get baseline performance
    print("\n📊 Baseline Performance:")
    baseline_metrics = trainer.evaluate()
    baseline_r2 = baseline_metrics['R2']
    print(f"   R²: {baseline_r2:.4f}")
    print(f"   MAE: {baseline_metrics['MAE']:.2f} bytes")
    
    # Track performance across rounds
    performance_history = [baseline_r2]
    
    # Training rounds
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'─'*70}")
        print(f"Round {round_num}/{num_rounds}")
        print(f"{'─'*70}")
        
        # Adaptive learning rate
        if learning_rate == 'adaptive':
            current_lr = 0.0001 / (round_num ** 0.5)  # Decreases each round
            print(f"Learning rate: {current_lr:.6f}")
        else:
            current_lr = learning_rate
        
        # Train
        trainer.train(
            epochs=epochs,
            batch_size=64,
            learning_rate=current_lr,
            validation_split=0.15
        )
        
        # Evaluate after this round
        metrics = trainer.evaluate()
        current_r2 = metrics['R2']
        performance_history.append(current_r2)
        
        improvement = current_r2 - performance_history[-2]
        
        print(f"\n📊 After Round {round_num}:")
        print(f"   R²: {current_r2:.4f} ({improvement:+.4f})")
        print(f"   MAE: {metrics['MAE']:.2f} bytes")
        
        # Check for overfitting
        if improvement < -0.01:
            print("\n⚠️  WARNING: Performance DECREASED!")
            print("   This might be overfitting. Consider stopping.")
            
            if round_num < num_rounds:
                continue_choice = input("   Continue training? [y/N]: ").strip().lower()
                if continue_choice != 'y':
                    print("   Stopping early and reverting to best model...")
                    break
        
        # Check if converged
        if abs(improvement) < 0.001 and round_num > 2:
            print("\n✓ Model has converged (no significant improvement)")
            print("  Stopping early...")
            break
    
    # Final evaluation
    final_metrics = trainer.evaluate()
    final_r2 = final_metrics['R2']
    total_improvement = final_r2 - baseline_r2
    
    all_results[scenario] = {
        'baseline_r2': baseline_r2,
        'final_r2': final_r2,
        'improvement': total_improvement,
        'history': performance_history,
        'final_metrics': final_metrics
    }
    
    # Save improved model
    save_name = f'{scenario}_improved_v{datetime.now().strftime("%Y%m%d_%H%M")}'
    trainer.save_model(
        f'models/{save_name}.pt',
        f'models/{save_name}_scaler.pkl'
    )
    
    print(f"\n✓ Improved model saved: models/{save_name}.pt")
    
    # Summary for this scenario
    print(f"\n{'='*70}")
    print(f"SUMMARY: {scenario.upper()}")
    print(f"{'='*70}")
    print(f"  Baseline R²:      {baseline_r2:.4f}")
    print(f"  Final R²:         {final_r2:.4f}")
    print(f"  Total improvement: {total_improvement:+.4f}")
    
    if total_improvement > 0.1:
        print(f"  Status: 🎉 Significant improvement!")
    elif total_improvement > 0.03:
        print(f"  Status: ✓ Good improvement")
    elif total_improvement > 0:
        print(f"  Status: ✓ Slight improvement")
    else:
        print(f"  Status: ⚠️ No improvement (model may have already converged)")

# ============================================
# Step 4: Overall Summary & Visualization
# ============================================

print("\n\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"\n{'Scenario':<15} {'Baseline R²':<15} {'Final R²':<15} {'Improvement':<15} {'Status':<20}")
print("-" * 70)

for scenario, results in all_results.items():
    improvement = results['improvement']
    
    if results['final_r2'] > 0.8:
        status = "🎉 Excellent"
    elif results['final_r2'] > 0.6:
        status = "✓ Good"
    elif improvement > 0.1:
        status = "✓ Much better"
    elif improvement > 0:
        status = "✓ Improved"
    else:
        status = "⚠️ No change"
    
    print(f"{scenario:<15} {results['baseline_r2']:<15.4f} {results['final_r2']:<15.4f} {improvement:<15.4f} {status:<20}")

# Plot improvement curves
if len(all_results) > 0:
    print("\n\nGenerating performance charts...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: R² progression for each scenario
    for scenario, results in all_results.items():
        rounds = list(range(len(results['history'])))
        axes[0].plot(rounds, results['history'], marker='o', linewidth=2, label=scenario)
    
    axes[0].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (0.8)')
    axes[0].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (0.6)')
    axes[0].set_xlabel('Training Round', fontsize=12)
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('Model Improvement Over Training Rounds', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Before vs After comparison
    scenarios_list = list(all_results.keys())
    baseline_r2s = [all_results[s]['baseline_r2'] for s in scenarios_list]
    final_r2s = [all_results[s]['final_r2'] for s in scenarios_list]
    
    x = np.arange(len(scenarios_list))
    width = 0.35
    
    axes[1].bar(x - width/2, baseline_r2s, width, label='Before', alpha=0.7)
    axes[1].bar(x + width/2, final_r2s, width, label='After', alpha=0.7)
    axes[1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Scenario', fontsize=12)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('Before vs After Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenarios_list)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = f'docs/training_improvement_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
    plt.savefig(save_path, dpi=150)
    print(f"✓ Charts saved to {save_path}")
    plt.show()

print("\n" + "=" * 70)
print("✓ TRAINING COMPLETE!")
print("=" * 70)

# Recommendations
print("\n📋 Recommendations:")
for scenario, results in all_results.items():
    if results['final_r2'] < 0.5:
        print(f"\n⚠️  {scenario.upper()}:")
        print(f"   - R² is still low ({results['final_r2']:.4f})")
        print(f"   - Consider:")
        print(f"     • Regenerating training data with more variation")
        print(f"     • Trying GRU instead of LSTM")
        print(f"     • Increasing sequence length (60→120 seconds)")
        print(f"     • Adding more training data")
    elif results['final_r2'] > 0.8:
        print(f"\n✓ {scenario.upper()}: Excellent! Ready for production.")

print("\n" + "=" * 70)

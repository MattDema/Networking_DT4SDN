# train_all_scenarios_v2.py
"""
A100-40GB Training Script v2 - Multi-file Support + Visualization
Trains 5 separate models (one per scenario) with comprehensive visualizations

Features:
- Combines multiple CSV files per scenario automatically
- Generates training history plots for each model
- Creates model comparison visualization
- Saves both .pt models and visualization images
"""

import torch
import sys
import os
import time
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, 'src')

from ml_models.model_trainer import TrafficModelTrainer

print("="*80)
print("🚀 A100-40GB TRAINING v2 - Multi-File + Visualization 🚀")
print("="*80)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# GPU VERIFICATION
# =============================================================================
if not torch.cuda.is_available():
    print("\n❌ ERROR: No GPU detected! Make sure you selected A100 in Colab.")
    print("   Runtime → Change runtime type → Hardware accelerator → A100 GPU")
    exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"\n✓ GPU: {gpu_name}")
print(f"✓ Memory: {gpu_memory:.1f} GB")
print(f"✓ CUDA Version: {torch.version.cuda}")
print(f"✓ PyTorch Version: {torch.__version__}")

# Enable A100 optimizations
torch.backends.cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')
    print("✓ Tensor Core precision: HIGH")

# =============================================================================
# CONFIGURATION
# =============================================================================
A100_CONFIG = {
    'sequence_length': 120,       # 2 minutes of history
    'prediction_horizon': 60,     # Predict 1 minute ahead
    'batch_size': 512,            # Large batch for A100
    'epochs': 2000,               # Max epochs (early stopping will kick in)
    'learning_rate': 0.0003,
    'patience': 200,              # Early stopping patience
    'units': [256, 256, 128, 64], # 4-layer deep network
    'dropout': 0.3,
    'validation_split': 0.15
}

# Scenarios and their data patterns
SCENARIOS = {
    'normal': 'network_data_normal_50000_*.csv',
    'burst': 'network_data_burst_50000_*.csv', 
    'congestion': 'network_data_congestion_50000_*.csv',
    'ddos': 'network_data_ddos_50000_*.csv',
    'mixed': 'network_data_mixed_50000_*.csv'
}

# Data folder path (relative to repo root)
DATA_FOLDER = 'src/ml_models/data_collection'

# Output folders
MODELS_FOLDER = 'models'
VISUALS_FOLDER = 'visuals/training'

# Create output folders
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(VISUALS_FOLDER, exist_ok=True)

print("\n" + "="*80)
print("⚡ A100 TRAINING CONFIGURATION ⚡")
print("="*80)
print(f"  Sequence Length:      {A100_CONFIG['sequence_length']}s")
print(f"  Prediction Horizon:   {A100_CONFIG['prediction_horizon']}s")
print(f"  Batch Size:           {A100_CONFIG['batch_size']}")
print(f"  Network Architecture: {A100_CONFIG['units']}")
print(f"  Network Depth:        {len(A100_CONFIG['units'])} layers")
print(f"  Max Epochs:           {A100_CONFIG['epochs']}")
print(f"  Early Stop Patience:  {A100_CONFIG['patience']}")
print(f"  Dropout:              {A100_CONFIG['dropout']}")
print("="*80)

# =============================================================================
# DATA LOADING HELPER
# =============================================================================
def load_multiple_csvs(scenario_name: str, pattern: str, data_folder: str) -> pd.DataFrame:
    """
    Load and combine multiple CSV files for a scenario
    
    Args:
        scenario_name: Name of the scenario (for logging)
        pattern: Glob pattern to match files (e.g., 'network_data_normal_*.csv')
        data_folder: Folder containing the CSV files
    
    Returns:
        Combined DataFrame with all data
    """
    full_pattern = os.path.join(data_folder, pattern)
    files = sorted(glob.glob(full_pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found matching: {full_pattern}")
    
    print(f"\n  📂 Found {len(files)} files for {scenario_name}:")
    
    dfs = []
    total_rows = 0
    for f in files:
        df = pd.read_csv(f)
        rows = len(df)
        total_rows += rows
        print(f"     - {os.path.basename(f)}: {rows:,} records")
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Reset timestamp to be sequential
    combined['timestamp'] = range(1, len(combined) + 1)
    
    print(f"  ✓ Combined: {total_rows:,} total records")
    
    return combined

# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================
def plot_training_history(train_losses: list, val_losses: list, 
                          scenario: str, metrics: dict, 
                          save_path: str) -> None:
    """
    Create a comprehensive training visualization for a single model
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        scenario: Scenario name
        metrics: Dict with R2, MAE, RMSE
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training Results: {scenario.upper()} Scenario', fontsize=14, fontweight='bold')
    
    # Plot 1: Training & Validation Loss
    ax1 = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=1.5, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (MSE)', fontsize=11)
    ax1.set_title('Training History', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Add best epoch marker
    best_epoch = val_losses.index(min(val_losses)) + 1
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, 
                label=f'Best: Epoch {best_epoch}')
    ax1.scatter([best_epoch], [min(val_losses)], color='green', s=100, zorder=5)
    
    # Plot 2: Metrics Summary
    ax2 = axes[1]
    metric_names = ['R² Score', 'MAE (bytes)', 'RMSE (bytes)']
    metric_values = [metrics['R2'], metrics['MAE'], metrics['RMSE']]
    
    # Create bar colors based on R2 score
    r2 = metrics['R2']
    if r2 > 0.9:
        bar_color = '#2ecc71'  # Green
        status = "EXCELLENT"
    elif r2 > 0.8:
        bar_color = '#27ae60'  # Dark Green
        status = "VERY GOOD"
    elif r2 > 0.7:
        bar_color = '#f39c12'  # Orange
        status = "GOOD"
    elif r2 > 0.5:
        bar_color = '#e67e22'  # Dark Orange
        status = "ACCEPTABLE"
    else:
        bar_color = '#e74c3c'  # Red
        status = "POOR"
    
    # Create text summary
    ax2.axis('off')
    summary_text = f"""
    MODEL PERFORMANCE SUMMARY
    ═══════════════════════════════════════
    
    📊 R² Score:       {metrics['R2']:.4f}
    📉 MAE:            {metrics['MAE']:,.2f} bytes
    📈 RMSE:           {metrics['RMSE']:,.2f} bytes
    
    ⏱️  Epochs Trained: {len(train_losses)}
    🎯 Best Epoch:     {best_epoch}
    📉 Final Val Loss: {val_losses[-1]:.6f}
    📉 Best Val Loss:  {min(val_losses):.6f}
    
    ═══════════════════════════════════════
    STATUS: {status}
    """
    
    ax2.text(0.5, 0.5, summary_text, transform=ax2.transAxes,
             fontsize=12, family='monospace', verticalalignment='center',
             horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor=bar_color, alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training visualization saved: {save_path}")

def plot_all_models_comparison(results: dict, save_path: str) -> None:
    """
    Create a comparison visualization of all trained models
    
    Args:
        results: Dict with results for each scenario
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ALL MODELS COMPARISON - A100 Training Results', fontsize=14, fontweight='bold')
    
    scenarios = list(results.keys())
    
    # Colors based on R2 score
    def get_color(r2):
        if r2 > 0.9: return '#2ecc71'
        elif r2 > 0.8: return '#27ae60'
        elif r2 > 0.7: return '#f39c12'
        elif r2 > 0.5: return '#e67e22'
        else: return '#e74c3c'
    
    colors = [get_color(results[s]['R2']) for s in scenarios]
    
    # Plot 1: R² Scores
    ax1 = axes[0, 0]
    r2_values = [results[s]['R2'] for s in scenarios]
    bars = ax1.bar(scenarios, r2_values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_title('Model Accuracy (R² Score)', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
    ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (>0.7)')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, r2_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: MAE Comparison
    ax2 = axes[0, 1]
    mae_values = [results[s]['MAE'] for s in scenarios]
    bars = ax2.bar(scenarios, mae_values, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('MAE (bytes)', fontsize=11)
    ax2.set_title('Mean Absolute Error', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.02,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Training Time
    ax3 = axes[1, 0]
    times = [results[s]['training_time_min'] for s in scenarios]
    bars = ax3.bar(scenarios, times, color='#3498db', edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Time (minutes)', fontsize=11)
    ax3.set_title('Training Time', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                 f'{val:.1f}m', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Epochs Trained
    ax4 = axes[1, 1]
    epochs = [results[s]['epochs_trained'] for s in scenarios]
    bars = ax4.bar(scenarios, epochs, color='#9b59b6', edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('Epochs', fontsize=11)
    ax4.set_title('Epochs Until Convergence', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, epochs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(epochs)*0.02,
                 f'{val}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ All models comparison saved: {save_path}")

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
results = {}
total_start = time.time()

print("\n" + "="*80)
print("📊 LOADING DATA FROM MULTIPLE FILES PER SCENARIO")
print("="*80)

for idx, (scenario, pattern) in enumerate(SCENARIOS.items(), 1):
    print(f"\n\n{'='*80}")
    print(f"[{idx}/{len(SCENARIOS)}] TRAINING: {scenario.upper()}")
    print('='*80)
    
    scenario_start = time.time()
    
    try:
        # Step 1: Load and combine data
        print(f"\n[1/5] Loading and combining data files...")
        combined_df = load_multiple_csvs(scenario, pattern, DATA_FOLDER)
        
        # Save combined data temporarily
        temp_csv = f'/tmp/combined_{scenario}.csv'
        combined_df.to_csv(temp_csv, index=False)
        
        # Step 2: Initialize trainer
        print(f"\n[2/5] Initializing trainer...")
        trainer = TrafficModelTrainer(
            sequence_length=A100_CONFIG['sequence_length'],
            prediction_horizon=A100_CONFIG['prediction_horizon'],
            model_type='lstm'
        )
        
        # Load combined data
        trainer.load_data(temp_csv, target_column='bytes')
        
        # Step 3: Build model
        print(f"\n[3/5] Building {len(A100_CONFIG['units'])}-layer BiLSTM model...")
        trainer.build_model(
            units=A100_CONFIG['units'],
            dropout=A100_CONFIG['dropout']
        )
        
        total_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"  ✓ Total parameters: {total_params:,}")
        print(f"  ✓ Estimated size: {total_params * 4 / 1e6:.1f} MB")
        
        # Step 4: Train
        print(f"\n[4/5] Training with A100 optimization...")
        print(f"      (Max {A100_CONFIG['epochs']} epochs, patience={A100_CONFIG['patience']})")
        
        training_start = time.time()
        
        trainer.train(
            epochs=A100_CONFIG['epochs'],
            batch_size=A100_CONFIG['batch_size'],
            learning_rate=A100_CONFIG['learning_rate'],
            validation_split=A100_CONFIG['validation_split']
        )
        
        training_time = time.time() - training_start
        
        # Step 5: Evaluate
        print(f"\n[5/5] Evaluating model...")
        metrics = trainer.evaluate()
        
        # Store results
        results[scenario] = {
            'R2': metrics['R2'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'training_time_min': training_time / 60,
            'total_params': total_params,
            'epochs_trained': len(trainer.train_losses),
            'train_losses': trainer.train_losses.copy(),
            'val_losses': trainer.val_losses.copy()
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"📊 RESULTS: {scenario.upper()}")
        print('='*60)
        print(f"  R² Score:       {metrics['R2']:.4f}")
        print(f"  MAE:            {metrics['MAE']:,.2f} bytes")
        print(f"  RMSE:           {metrics['RMSE']:,.2f} bytes")
        print(f"  Training Time:  {training_time/60:.1f} minutes")
        print(f"  Epochs:         {len(trainer.train_losses)}")
        
        # Status emoji
        r2 = metrics['R2']
        if r2 > 0.9:
            print(f"  Status:         🏆 EXCELLENT!")
        elif r2 > 0.8:
            print(f"  Status:         ✨ VERY GOOD!")
        elif r2 > 0.7:
            print(f"  Status:         ✓ GOOD")
        elif r2 > 0.5:
            print(f"  Status:         ⚠️ ACCEPTABLE")
        else:
            print(f"  Status:         ❌ POOR - needs work")
        print('='*60)
        
        # Save model
        model_path = f'{MODELS_FOLDER}/{scenario}_ultimate.pt'
        scaler_path = f'{MODELS_FOLDER}/{scenario}_ultimate_scaler.pkl'
        trainer.save_model(model_path, scaler_path)
        
        # Generate visualization for this model
        vis_path = f'{VISUALS_FOLDER}/{scenario}_training.png'
        plot_training_history(
            trainer.train_losses,
            trainer.val_losses,
            scenario,
            metrics,
            vis_path
        )
        
        # Cleanup temp file
        os.remove(temp_csv)
        
        scenario_time = time.time() - scenario_start
        print(f"\n✓ {scenario.upper()} complete in {scenario_time/60:.1f} minutes")
        
    except Exception as e:
        print(f"\n❌ ERROR training {scenario}:")
        import traceback
        traceback.print_exc()
        continue

# =============================================================================
# FINAL SUMMARY & COMPARISON VISUALIZATION
# =============================================================================
total_time = time.time() - total_start

print("\n\n" + "="*80)
print("🏁 TRAINING COMPLETE - FINAL SUMMARY 🏁")
print("="*80)

print(f"\n{'Scenario':<12} {'R²':<10} {'MAE':<12} {'Time(min)':<10} {'Epochs':<8} {'Status'}")
print("-"*80)

for scenario, m in results.items():
    status = "🏆" if m['R2'] > 0.9 else "✨" if m['R2'] > 0.8 else "✓" if m['R2'] > 0.7 else "⚠️" if m['R2'] > 0.5 else "❌"
    print(f"{scenario:<12} {m['R2']:<10.4f} {m['MAE']:<12,.0f} {m['training_time_min']:<10.1f} "
          f"{m['epochs_trained']:<8} {status}")

print("-"*80)
print(f"{'TOTAL TIME:':<12} {'':<10} {'':<12} {total_time/60:<10.1f}")
print("="*80)

# Generate comparison visualization
if results:
    comparison_path = f'{VISUALS_FOLDER}/all_models_comparison.png'
    plot_all_models_comparison(results, comparison_path)

# Final stats
if results:
    avg_r2 = np.mean([m['R2'] for m in results.values()])
    print(f"\n📊 Average R² Score: {avg_r2:.4f}")
    print(f"⏱️  Total Training Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"\n📁 Outputs:")
    print(f"   Models:  {MODELS_FOLDER}/*_ultimate.pt")
    print(f"   Scalers: {MODELS_FOLDER}/*_ultimate_scaler.pkl")
    print(f"   Visuals: {VISUALS_FOLDER}/*.png")

print(f"\n📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "="*80)
print("🚀 A100 TRAINING v2 COMPLETE! 🚀")
print("="*80)

# train_all_scenarios_v3.py
"""
IMPROVED Training Script - Fixes for Poor R² Scores

Key improvements:
1. Shorter sequence length (60→30) - traffic is highly variable
2. Simpler architecture (2 layers instead of 4)
3. Smaller prediction horizon (60→15) - more achievable
4. StandardScaler instead of MinMaxScaler - handles outliers better
5. Gradient clipping - prevents exploding gradients
6. Cosine annealing LR - better convergence
7. Data smoothing option - reduce noise
8. Larger patience - give model more time
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import time
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("="*80)
print("🚀 A100 TRAINING v3 - IMPROVED FOR NOISY DATA 🚀")
print("="*80)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# GPU Setup
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    print("⚠️ Running on CPU")

# =============================================================================
# IMPROVED CONFIGURATION - Key changes from v2
# =============================================================================
CONFIG = {
    # REDUCED complexity - more achievable predictions
    'sequence_length': 60,        # Was 120 → 30 (less history = less noise)
    'prediction_horizon': 30,     # Was 60 → 15 (easier to predict near future)
    
    # SIMPLER architecture - less overfitting risk
    'units': [256, 128, 64],           # Was [256, 256, 128, 64] → [128, 64]
    'dropout': 0.2,               # Was 0.3 → 0.2 (less regularization for simple model)
    'bidirectional': True,        # Keep bidirectional
    
    # Training parameters
    'batch_size': 128,            # Was 512 → 128 (more gradient updates)
    'epochs': 500,                # Was 2000 → 500 (enough with better config)
    'learning_rate': 0.001,       # Was 0.0003 → 0.001 (faster initial learning)
    'patience': 200,              # Was 200 → 100 (still generous)
    'validation_split': 0.15,
    
    # New options
    'use_robust_scaler': True,    # Handles outliers better than MinMax
    'smooth_data': True,          # Apply rolling average to reduce noise
    'smooth_window': 5,           # Rolling window size
    'gradient_clip': 1.0,         # Gradient clipping
}

# Scenarios
SCENARIOS = {
    'normal': 'network_data_normal_50000_*.csv',
    'burst': 'network_data_burst_50000_*.csv', 
    'congestion': 'network_data_congestion_50000_*.csv',
    'ddos': 'network_data_ddos_50000_*.csv',
    'mixed': 'network_data_mixed_50000_*.csv'
}

DATA_FOLDER = 'src/ml_models/data_collection'
MODELS_FOLDER = 'models'
VISUALS_FOLDER = 'visuals/training'
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(VISUALS_FOLDER, exist_ok=True)

print("\n" + "="*80)
print("⚡ IMPROVED CONFIGURATION (v3) ⚡")
print("="*80)
print(f"  Sequence Length:    {CONFIG['sequence_length']}s (shorter = less noise)")
print(f"  Prediction Horizon: {CONFIG['prediction_horizon']}s (easier target)")
print(f"  Architecture:       {CONFIG['units']} (simpler)")
print(f"  Batch Size:         {CONFIG['batch_size']}")
print(f"  Learning Rate:      {CONFIG['learning_rate']}")
print(f"  Robust Scaler:      {CONFIG['use_robust_scaler']} (handles outliers)")
print(f"  Data Smoothing:     {CONFIG['smooth_data']} (window={CONFIG['smooth_window']})")
print("="*80)

# =============================================================================
# SIMPLER LSTM MODEL
# =============================================================================
class SimpleLSTM(nn.Module):
    """Simpler LSTM - less prone to overfitting on noisy data"""
    
    def __init__(self, input_size, hidden_units, prediction_horizon, dropout=0.2, bidirectional=True):
        super().__init__()
        
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        
        # Single stacked LSTM (more efficient than ModuleList)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units[0],
            num_layers=len(hidden_units),
            batch_first=True,
            dropout=dropout if len(hidden_units) > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        final_size = hidden_units[0] * num_directions
        self.fc = nn.Sequential(
            nn.Linear(final_size, hidden_units[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units[-1], prediction_horizon)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Take last timestep
        out = self.dropout(last_output)
        out = self.fc(out)
        return out.unsqueeze(-1)

# =============================================================================
# DATA LOADING WITH SMOOTHING
# =============================================================================
def load_and_preprocess(scenario, pattern, data_folder, config):
    """Load data with optional smoothing"""
    
    full_pattern = os.path.join(data_folder, pattern)
    files = sorted(glob.glob(full_pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found: {full_pattern}")
    
    print(f"\n  📂 Loading {len(files)} files...")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        print(f"     - {os.path.basename(f)}: {len(df):,} rows")
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    data = combined['bytes'].values.astype(np.float32)
    
    print(f"  ✓ Total samples: {len(data):,}")
    print(f"  📊 Range: [{data.min():,.0f}, {data.max():,.0f}] bytes")
    print(f"  📊 Mean: {data.mean():,.0f}, Std: {data.std():,.0f}")
    
    # Optional smoothing
    if config['smooth_data']:
        window = config['smooth_window']
        data = pd.Series(data).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        print(f"  ✓ Applied smoothing (window={window})")
    
    return data.reshape(-1, 1)

def create_sequences(data, seq_len, pred_horizon):
    """Create input-output sequences"""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_horizon):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_horizon])
    return np.array(X), np.array(y)

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_model(scenario, data, config):
    """Train a single model with improved settings"""
    
    seq_len = config['sequence_length']
    pred_horizon = config['prediction_horizon']
    
    # Normalize with RobustScaler (handles outliers better!)
    if config['use_robust_scaler']:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_len, pred_horizon)
    print(f"  ✓ Created {len(X):,} sequences")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    val_size = int(len(X_train) * config['validation_split'])
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    
    print(f"  📊 Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # DataLoaders
    train_loader = DataLoader(TrafficDataset(X_train, y_train), 
                              batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(TrafficDataset(X_val, y_val), 
                            batch_size=config['batch_size'])
    test_loader = DataLoader(TrafficDataset(X_test, y_test), 
                             batch_size=config['batch_size'])
    
    # Model
    model = SimpleLSTM(
        input_size=1,
        hidden_units=config['units'],
        prediction_horizon=pred_horizon,
        dropout=config['dropout'],
        bidirectional=config['bidirectional']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model params: {total_params:,}")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    best_state = None
    
    print(f"\n  🏋️ Training (max {config['epochs']} epochs)...")
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 25 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"     Epoch {epoch+1}/{config['epochs']} - "
                  f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {lr:.2e}")
        
        if patience_counter >= config['patience']:
            print(f"  ⏹️ Early stop at epoch {epoch+1}")
            break
    
    # Restore best model
    model.load_state_dict(best_state)
    
    # Evaluate on test set
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    y_pred = np.concatenate(all_preds).reshape(-1, 1)
    y_true = np.concatenate(all_targets).reshape(-1, 1)
    
    # Denormalize
    y_pred_denorm = scaler.inverse_transform(y_pred)
    y_true_denorm = scaler.inverse_transform(y_true)
    
    # Metrics
    mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
    rmse = np.sqrt(mean_squared_error(y_true_denorm, y_pred_denorm))
    r2 = r2_score(y_true_denorm, y_pred_denorm)
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': {'R2': r2, 'MAE': mae, 'RMSE': rmse},
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_trained': len(train_losses)
    }

# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_results(scenario, result, save_path):
    """Plot training results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{scenario.upper()} - Training Results', fontsize=14, fontweight='bold')
    
    metrics = result['metrics']
    train_losses = result['train_losses']
    val_losses = result['val_losses']
    
    # Loss plot
    ax1 = axes[0]
    ax1.plot(train_losses, 'b-', label='Train', alpha=0.7)
    ax1.plot(val_losses, 'r-', label='Validation', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Metrics
    ax2 = axes[1]
    ax2.axis('off')
    
    r2 = metrics['R2']
    color = '#2ecc71' if r2 > 0.8 else '#f39c12' if r2 > 0.5 else '#e74c3c'
    status = '🏆 EXCELLENT' if r2 > 0.8 else '✓ GOOD' if r2 > 0.5 else '❌ POOR'
    
    text = f"""
    ═══════════════════════════════
           MODEL PERFORMANCE
    ═══════════════════════════════
    
    R² Score:    {metrics['R2']:.4f}
    MAE:         {metrics['MAE']:,.0f} bytes
    RMSE:        {metrics['RMSE']:,.0f} bytes
    
    Epochs:      {result['epochs_trained']}
    
    Status:      {status}
    ═══════════════════════════════
    """
    
    ax2.text(0.5, 0.5, text, transform=ax2.transAxes, fontsize=12,
             family='monospace', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
results = {}
total_start = time.time()

for idx, (scenario, pattern) in enumerate(SCENARIOS.items(), 1):
    print(f"\n\n{'='*80}")
    print(f"[{idx}/{len(SCENARIOS)}] TRAINING: {scenario.upper()}")
    print('='*80)
    
    try:
        # Load data
        data = load_and_preprocess(scenario, pattern, DATA_FOLDER, CONFIG)
        
        # Train
        scenario_start = time.time()
        result = train_model(scenario, data, CONFIG)
        training_time = time.time() - scenario_start
        
        # Results
        metrics = result['metrics']
        print(f"\n  📊 RESULTS:")
        print(f"     R² Score: {metrics['R2']:.4f}")
        print(f"     MAE:      {metrics['MAE']:,.0f} bytes")
        print(f"     RMSE:     {metrics['RMSE']:,.0f} bytes")
        print(f"     Time:     {training_time/60:.1f} min")
        
        # Save model
        model_path = f"{MODELS_FOLDER}/{scenario}_v3.pt"
        scaler_path = f"{MODELS_FOLDER}/{scenario}_v3_scaler.pkl"
        
        torch.save({
            'model': result['model'],
            'model_state_dict': result['model'].state_dict(),
            'config': CONFIG
        }, model_path)
        joblib.dump(result['scaler'], scaler_path)
        
        # Save visualization
        vis_path = f"{VISUALS_FOLDER}/{scenario}_v3_training.png"
        plot_results(scenario, result, vis_path)
        
        results[scenario] = {
            **metrics,
            'training_time_min': training_time / 60,
            'epochs_trained': result['epochs_trained']
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# Final summary
print("\n\n" + "="*80)
print("🏁 FINAL SUMMARY")
print("="*80)

if results:
    print(f"\n{'Scenario':<12} {'R²':<10} {'MAE':<12} {'Time':<8} {'Status'}")
    print("-"*60)
    for s, m in results.items():
        status = "🏆" if m['R2'] > 0.8 else "✓" if m['R2'] > 0.5 else "❌"
        print(f"{s:<12} {m['R2']:<10.4f} {m['MAE']:<12,.0f} {m['training_time_min']:<8.1f} {status}")
    
    avg_r2 = np.mean([m['R2'] for m in results.values()])
    print("-"*60)
    print(f"{'Average R²:':<12} {avg_r2:.4f}")

print(f"\n⏱️ Total time: {(time.time()-total_start)/60:.1f} minutes")
print("="*80)

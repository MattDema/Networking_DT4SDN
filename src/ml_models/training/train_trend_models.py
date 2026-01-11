# train_trend_models.py
"""
TREND PREDICTION Training Script - Predicts smoothed traffic trends

Key differences from previous versions:
- Heavy smoothing (window=20) - removes noise, keeps trends
- Predicts TRENDS, not exact values
- 90 second prediction horizon
- Deep architecture for complex patterns
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
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("="*80)
print("🎯 TREND PREDICTION MODEL - Heavy Smoothing + Deep Architecture")
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
# ROBUST TREND PREDICTION CONFIG
# =============================================================================
CONFIG = {
    # LONGER history and prediction for trend analysis
    'sequence_length': 120,       # 2 minutes of history (plenty of context)
    'prediction_horizon': 90,     # Predict 90 seconds ahead (as requested!)
    
    # DEEP architecture - we can afford this with smoothed data
    'units': [256, 256, 128, 64], # 4-layer deep network
    'dropout': 0.25,              # Moderate dropout
    'bidirectional': True,        # Bidirectional LSTM
    
    # AGGRESSIVE smoothing - THE KEY TO SUCCESS
    'smooth_window': 20,          # Heavy smoothing (20 samples)
    
    # Training parameters
    'batch_size': 256,            # Large batch for stable gradients
    'epochs': 1000,               # More epochs - early stopping will handle it
    'learning_rate': 0.0005,      # Moderate learning rate
    'patience': 150,              # Generous patience
    'validation_split': 0.15,
    'gradient_clip': 1.0,
    
    # Scheduler
    'use_cosine_annealing': True,
    'warmup_epochs': 10,
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
print("⚡ TREND PREDICTION CONFIGURATION ⚡")
print("="*80)
print(f"  📊 Sequence Length:    {CONFIG['sequence_length']}s (2 min history)")
print(f"  🎯 Prediction Horizon: {CONFIG['prediction_horizon']}s (90s ahead!)")
print(f"  🏗️  Architecture:       {CONFIG['units']} (4 deep layers)")
print(f"  🔄 Bidirectional:      {CONFIG['bidirectional']}")
print(f"  📉 Smoothing Window:   {CONFIG['smooth_window']} samples (HEAVY)")
print(f"  📦 Batch Size:         {CONFIG['batch_size']}")
print(f"  ⏱️  Max Epochs:         {CONFIG['epochs']}")
print(f"  🛑 Early Stop:         {CONFIG['patience']} patience")
print("="*80)

# =============================================================================
# DEEP BILSTM MODEL FOR TRENDS
# =============================================================================
class TrendLSTM(nn.Module):
    """Deep BiLSTM for trend prediction"""
    
    def __init__(self, input_size, hidden_units, prediction_horizon, dropout=0.25, bidirectional=True):
        super().__init__()
        
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        
        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units[0],
            num_layers=len(hidden_units),
            batch_first=True,
            dropout=dropout if len(hidden_units) > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Batch normalization for stability
        lstm_output_size = hidden_units[0] * num_directions
        self.bn = nn.BatchNorm1d(lstm_output_size)
        
        # Attention mechanism (simple)
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 4),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 4, 1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_units[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units[-1], hidden_units[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_units[-1] // 2, prediction_horizon)
        )
    
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum across time
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
        
        # Batch normalization
        context = self.bn(context)
        
        # Output
        out = self.fc(context)  # (batch, prediction_horizon)
        return out.unsqueeze(-1)

# =============================================================================
# DATA LOADING WITH HEAVY SMOOTHING
# =============================================================================
def load_and_smooth(scenario, pattern, data_folder, smooth_window):
    """Load data and apply HEAVY smoothing for trend extraction"""
    
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
    raw_data = combined['bytes'].values.astype(np.float32)
    
    print(f"  ✓ Total samples: {len(raw_data):,}")
    print(f"  📊 Raw range: [{raw_data.min():,.0f}, {raw_data.max():,.0f}]")
    
    # HEAVY SMOOTHING - this is the key!
    smoothed = pd.Series(raw_data).rolling(
        window=smooth_window, 
        center=True,
        min_periods=1
    ).mean().values
    
    # Additional exponential smoothing for ultra-smooth trends
    alpha = 0.3
    ema = pd.Series(smoothed).ewm(alpha=alpha, adjust=False).mean().values
    
    print(f"  ✓ Applied smoothing: rolling({smooth_window}) + EMA(α={alpha})")
    print(f"  📊 Smoothed range: [{ema.min():,.0f}, {ema.max():,.0f}]")
    
    return ema.reshape(-1, 1)

def create_sequences(data, seq_len, pred_horizon):
    """Create input-output sequences"""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_horizon):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_horizon])
    return np.array(X), np.array(y)

class TrendDataset(Dataset):
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
def train_trend_model(scenario, data, config):
    """Train a trend prediction model"""
    
    seq_len = config['sequence_length']
    pred_horizon = config['prediction_horizon']
    
    # Robust scaler
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_len, pred_horizon)
    print(f"  ✓ Created {len(X):,} sequences")
    
    # Split (time-aware)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    val_size = int(len(X_train) * config['validation_split'])
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    
    print(f"  📊 Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # DataLoaders
    train_loader = DataLoader(TrendDataset(X_train, y_train), 
                              batch_size=config['batch_size'], shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(TrendDataset(X_val, y_val), 
                            batch_size=config['batch_size'],
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(TrendDataset(X_test, y_test), 
                             batch_size=config['batch_size'],
                             num_workers=0, pin_memory=True)
    
    # Model
    model = TrendLSTM(
        input_size=1,
        hidden_units=config['units'],
        prediction_horizon=pred_horizon,
        dropout=config['dropout'],
        bidirectional=config['bidirectional']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model params: {total_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    
    # Scheduler: cosine annealing with warm restarts
    if config['use_cosine_annealing']:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
    
    # Loss function: Huber loss (more robust to outliers)
    criterion = nn.SmoothL1Loss()
    
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
        
        # Scheduler step
        if config['use_cosine_annealing']:
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch < 5:
            lr = optimizer.param_groups[0]['lr']
            print(f"     Epoch {epoch+1:4d}/{config['epochs']} - "
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
        'epochs_trained': len(train_losses),
        'config': config
    }

# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_trend_results(scenario, result, save_path):
    """Plot training results for trend model"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'TREND MODEL: {scenario.upper()} - 90s Prediction', fontsize=14, fontweight='bold')
    
    metrics = result['metrics']
    train_losses = result['train_losses']
    val_losses = result['val_losses']
    
    # Loss plot
    ax1 = axes[0]
    ax1.plot(train_losses, 'b-', label='Train', alpha=0.7, linewidth=1.5)
    ax1.plot(val_losses, 'r-', label='Validation', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Smooth L1)')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Mark best epoch
    best_epoch = val_losses.index(min(val_losses))
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best: {best_epoch}')
    
    # Metrics panel
    ax2 = axes[1]
    ax2.axis('off')
    
    r2 = metrics['R2']
    if r2 > 0.85:
        color = '#27ae60'
        status = '🏆 EXCELLENT'
    elif r2 > 0.7:
        color = '#2ecc71'
        status = '✨ VERY GOOD'
    elif r2 > 0.5:
        color = '#f39c12'
        status = '✓ GOOD'
    else:
        color = '#e74c3c'
        status = '❌ POOR'
    
    text = f"""
    ════════════════════════════════════
         TREND PREDICTION RESULTS
    ════════════════════════════════════
    
    📊 R² Score:       {metrics['R2']:.4f}
    📉 MAE:            {metrics['MAE']:,.0f} bytes
    📈 RMSE:           {metrics['RMSE']:,.0f} bytes
    
    ⏱️  Epochs:         {result['epochs_trained']}
    🎯 Best Epoch:     {best_epoch}
    
    🔮 Predicts:       {result['config']['prediction_horizon']}s ahead
    📜 History:        {result['config']['sequence_length']}s
    
    ════════════════════════════════════
    STATUS: {status}
    ════════════════════════════════════
    """
    
    ax2.text(0.5, 0.5, text, transform=ax2.transAxes, fontsize=11,
             family='monospace', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_comparison(results, save_path):
    """Compare all models"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = list(results.keys())
    r2_scores = [results[s]['R2'] for s in scenarios]
    
    colors = ['#27ae60' if r > 0.7 else '#f39c12' if r > 0.5 else '#e74c3c' for r in r2_scores]
    
    bars = ax.bar(scenarios, r2_scores, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='OK (0.5)')
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('TREND PREDICTION - All Scenarios (90s Horizon)', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.2, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# =============================================================================
# MAIN TRAINING
# =============================================================================
results = {}
total_start = time.time()

for idx, (scenario, pattern) in enumerate(SCENARIOS.items(), 1):
    print(f"\n\n{'='*80}")
    print(f"[{idx}/{len(SCENARIOS)}] TRAINING TREND MODEL: {scenario.upper()}")
    print('='*80)
    
    try:
        # Load and smooth data
        data = load_and_smooth(scenario, pattern, DATA_FOLDER, CONFIG['smooth_window'])
        
        # Train
        scenario_start = time.time()
        result = train_trend_model(scenario, data, CONFIG)
        training_time = time.time() - scenario_start
        
        # Results
        metrics = result['metrics']
        print(f"\n  📊 RESULTS:")
        print(f"     R² Score: {metrics['R2']:.4f}")
        print(f"     MAE:      {metrics['MAE']:,.0f} bytes")
        print(f"     RMSE:     {metrics['RMSE']:,.0f} bytes")
        print(f"     Time:     {training_time/60:.1f} min")
        
        r2 = metrics['R2']
        if r2 > 0.85:
            print(f"     Status:   🏆 EXCELLENT!")
        elif r2 > 0.7:
            print(f"     Status:   ✨ VERY GOOD!")
        elif r2 > 0.5:
            print(f"     Status:   ✓ GOOD")
        else:
            print(f"     Status:   ⚠️ Needs work")
        
        # Save model
        model_path = f"{MODELS_FOLDER}/{scenario}_trend.pt"
        scaler_path = f"{MODELS_FOLDER}/{scenario}_trend_scaler.pkl"
        
        torch.save({
            'model': result['model'],
            'model_state_dict': result['model'].state_dict(),
            'config': {
                'sequence_length': CONFIG['sequence_length'],
                'prediction_horizon': CONFIG['prediction_horizon'],
                'num_features': 1,
                'smooth_window': CONFIG['smooth_window'],
                'model_type': 'TrendLSTM'
            }
        }, model_path)
        joblib.dump(result['scaler'], scaler_path)
        print(f"  ✓ Saved: {model_path}")
        
        # Save visualization
        vis_path = f"{VISUALS_FOLDER}/{scenario}_trend_training.png"
        plot_trend_results(scenario, result, vis_path)
        print(f"  ✓ Plot: {vis_path}")
        
        results[scenario] = {
            **metrics,
            'training_time_min': training_time / 60,
            'epochs_trained': result['epochs_trained']
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# Comparison plot
if results:
    comparison_path = f"{VISUALS_FOLDER}/all_trend_models_comparison.png"
    plot_comparison(results, comparison_path)
    print(f"\n✓ Comparison: {comparison_path}")

# Final summary
print("\n\n" + "="*80)
print("🏁 TREND PREDICTION TRAINING COMPLETE")
print("="*80)

if results:
    print(f"\n{'Scenario':<12} {'R²':<10} {'MAE':<12} {'Time':<8} {'Status'}")
    print("-"*60)
    for s, m in results.items():
        status = "🏆" if m['R2'] > 0.85 else "✨" if m['R2'] > 0.7 else "✓" if m['R2'] > 0.5 else "⚠️"
        print(f"{s:<12} {m['R2']:<10.4f} {m['MAE']:<12,.0f} {m['training_time_min']:<8.1f} {status}")
    
    avg_r2 = np.mean([m['R2'] for m in results.values()])
    print("-"*60)
    print(f"{'Average R²:':<12} {avg_r2:.4f}")
    
    good_count = sum(1 for m in results.values() if m['R2'] > 0.7)
    print(f"\n🎯 Good models (R² > 0.7): {good_count}/{len(results)}")

print(f"\n⏱️ Total time: {(time.time()-total_start)/60:.1f} minutes")
print(f"\n📁 Models saved to: {MODELS_FOLDER}/*_trend.pt")
print(f"📊 Plots saved to: {VISUALS_FOLDER}/*_trend_training.png")
print("="*80)

# train_seq2seq_predictor.py
"""
FAST SEQ2SEQ TRAFFIC PREDICTOR - RTX 3050 OPTIMIZED

Simple LSTM encoder-decoder WITHOUT attention (much faster!):
- Input: 90 seconds of history
- Output: 60 future traffic values
- Used alongside classifier for realistic graph predictions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
import gc
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 FAST SEQ2SEQ TRAFFIC PREDICTOR - RTX 3050 🚀")
print("="*80)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# GPU SETUP
# =============================================================================
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n🎮 GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    print("⚠️ Running on CPU")

# =============================================================================
# FAST CONFIG - Optimized for speed
# =============================================================================
CONFIG = {
    'sequence_length': 90,
    'prediction_horizon': 60,
    
    # SIMPLIFIED architecture (no attention = much faster)
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    
    # Larger batch = faster
    'batch_size': 256,
    
    # Training
    'epochs': 1000,
    'learning_rate': 0.002,
    'patience': 30,
    'weight_decay': 1e-5,
    
    'smooth_window': 5,
}

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
print("⚡ FAST SEQ2SEQ CONFIG ⚡")
print("="*80)
print(f"  📊 Input:      {CONFIG['sequence_length']}s → Output: {CONFIG['prediction_horizon']} values")
print(f"  🧠 Hidden:     {CONFIG['hidden_size']} (BiLSTM, no attention)")
print(f"  📦 Batch:      {CONFIG['batch_size']}")
print(f"  ⏱️  Epochs:     {CONFIG['epochs']} max")
print("="*80)


# =============================================================================
# SIMPLE FAST MODEL (No Attention!)
# =============================================================================
class FastSeq2Seq(nn.Module):
    """
    Fast LSTM Encoder-Decoder without attention.
    Uses direct projection from encoder to decoder.
    """
    
    def __init__(self, config):
        super().__init__()
        hidden = config['hidden_size']
        layers = config['num_layers']
        dropout = config['dropout']
        
        # Encoder: BiLSTM
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
            bidirectional=True
        )
        
        # Bridge: Project encoder hidden to decoder hidden
        self.bridge_h = nn.Linear(hidden * 2, hidden)
        self.bridge_c = nn.Linear(hidden * 2, hidden)
        
        # Decoder: Unidirectional LSTM
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        
        # Output projection
        self.fc = nn.Linear(hidden, 1)
        
        self.prediction_horizon = config['prediction_horizon']
        self.num_layers = layers
        self.hidden_size = hidden
    
    def forward(self, src, target=None, teacher_forcing=0.5):
        batch_size = src.size(0)
        
        # Encode
        _, (h_enc, c_enc) = self.encoder(src)
        
        # Bridge: combine bidirectional states
        h_enc = h_enc.view(self.num_layers, 2, batch_size, self.hidden_size)
        c_enc = c_enc.view(self.num_layers, 2, batch_size, self.hidden_size)
        
        h_cat = torch.cat([h_enc[:, 0], h_enc[:, 1]], dim=2)  # (layers, batch, hidden*2)
        c_cat = torch.cat([c_enc[:, 0], c_enc[:, 1]], dim=2)
        
        h_dec = self.bridge_h(h_cat)  # (layers, batch, hidden)
        c_dec = self.bridge_c(c_cat)
        
        # Decode all at once (faster than loop!)
        # Create decoder input: last encoder value repeated
        last_val = src[:, -1:, :]  # (batch, 1, 1)
        decoder_input = last_val.repeat(1, self.prediction_horizon, 1)  # (batch, 60, 1)
        
        # Run decoder
        decoder_output, _ = self.decoder(decoder_input, (h_dec, c_dec))
        
        # Project to output
        output = self.fc(decoder_output).squeeze(-1)  # (batch, 60)
        
        return output


# =============================================================================
# DATASET
# =============================================================================
class Seq2SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(scenario, pattern, config):
    full_pattern = os.path.join(DATA_FOLDER, pattern)
    files = sorted(glob.glob(full_pattern))
    
    if not files:
        raise FileNotFoundError(f"No files: {full_pattern}")
    
    print(f"\n  📂 Loading {len(files)} files...")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"     - {os.path.basename(f)}: {len(df):,} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    data = combined['bytes'].values.astype(np.float32)
    
    # Smooth and clip
    data = pd.Series(data).rolling(config['smooth_window'], center=True, min_periods=1).mean().values
    data = np.clip(data, 0, None)
    
    print(f"  ✓ Total: {len(data):,} | Min: {data.min():.0f} | Mean: {data.mean():.0f} | Max: {data.max():.0f}")
    
    return data.reshape(-1, 1)


def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_len].flatten())
    return np.array(X), np.array(y)


# =============================================================================
# TRAINING
# =============================================================================
def train_model(scenario, data, config):
    seq_len = config['sequence_length']
    pred_len = config['prediction_horizon']
    
    # Normalize
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_len, pred_len)
    print(f"  ✓ Created {len(X):,} sequences")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    print(f"  📊 Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # DataLoaders
    train_loader = DataLoader(Seq2SeqDataset(X_train, y_train), 
                              batch_size=config['batch_size'], shuffle=True, 
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Seq2SeqDataset(X_val, y_val), 
                            batch_size=config['batch_size'], num_workers=0, pin_memory=True)
    test_loader = DataLoader(Seq2SeqDataset(X_test, y_test), 
                             batch_size=config['batch_size'], num_workers=0, pin_memory=True)
    
    # Model
    model = FastSeq2Seq(config).to(device)
    print(f"  ✓ Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val = float('inf')
    patience = 0
    train_losses, val_losses = [], []
    best_state = None
    
    print(f"\n  🏋️ Training...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            best_state = model.state_dict().copy()
        else:
            patience += 1
        
        # Progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]['lr']
            print(f"     Epoch {epoch+1:3d}/{config['epochs']} | "
                  f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                  f"LR: {lr:.1e} | {epoch_time:.1f}s")
        
        if patience >= config['patience']:
            print(f"  ⏹️ Early stop at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"  ✓ Training complete in {total_time/60:.1f} min")
    
    # Restore best
    model.load_state_dict(best_state)
    
    # Test evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # Cleanup
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return {
        'model': model,
        'scaler': scaler,
        'mse': mse,
        'mae': mae,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': len(train_losses),
        'predictions': all_preds,
        'targets': all_targets,
        'time_min': total_time / 60
    }


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_results(scenario, result, config, save_path):
    """Plot training results like the classifier does"""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'FAST SEQ2SEQ: {scenario.upper()} (RTX 3050)', fontsize=14, fontweight='bold')
    
    # 1. Training curves
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(result['train_losses'], 'b-', label='Train', alpha=0.7)
    ax1.plot(result['val_losses'], 'r-', label='Val', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Sample prediction
    ax2 = plt.subplot(2, 2, 2)
    idx = np.random.randint(0, len(result['predictions']))
    ax2.plot(range(60), result['targets'][idx], 'b-', label='Actual', linewidth=2)
    ax2.plot(range(60), result['predictions'][idx], 'r--', label='Predicted', linewidth=2)
    ax2.set_xlabel('Future second')
    ax2.set_ylabel('Traffic (normalized)')
    ax2.set_title('Sample Prediction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax3 = plt.subplot(2, 2, 3)
    errors = (result['predictions'] - result['targets']).flatten()
    ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Error Distribution (MAE: {result["mae"]:.4f})')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    mse, mae = result['mse'], result['mae']
    status = '🏆 EXCELLENT' if mse < 0.1 else '✨ GOOD' if mse < 0.3 else '⚠️ OK'
    color = '#27ae60' if mse < 0.1 else '#f39c12' if mse < 0.3 else '#e74c3c'
    
    text = f"""
    ═══════════════════════════════════
       SEQ2SEQ RESULTS
    ═══════════════════════════════════
    
    📊 MSE:      {mse:.6f}
    📈 MAE:      {mae:.6f}
    
    ⏱️  Epochs:   {result['epochs']}
    ⏱️  Time:     {result['time_min']:.1f} min
    
    🔮 Predicts: {config['prediction_horizon']} future values
    📜 History:  {config['sequence_length']}s
    
    ═══════════════════════════════════
    STATUS: {status}
    ═══════════════════════════════════
    """
    
    ax4.text(0.5, 0.5, text, transform=ax4.transAxes, fontsize=11,
             family='monospace', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
all_results = {}
total_start = time.time()

for idx, (scenario, pattern) in enumerate(SCENARIOS.items(), 1):
    print(f"\n\n{'='*80}")
    print(f"[{idx}/{len(SCENARIOS)}] TRAINING: {scenario.upper()}")
    print('='*80)
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        data = load_data(scenario, pattern, CONFIG)
        result = train_model(scenario, data, CONFIG)
        
        mse, mae = result['mse'], result['mae']
        print(f"\n  📊 RESULTS: MSE={mse:.6f}, MAE={mae:.6f}")
        
        # Save
        model_path = f"{MODELS_FOLDER}/{scenario}_seq2seq_3050.pt"
        torch.save({
            'model_state_dict': result['model'].state_dict(),
            'config': {
                'sequence_length': CONFIG['sequence_length'],
                'prediction_horizon': CONFIG['prediction_horizon'],
                'hidden_size': CONFIG['hidden_size'],
                'num_layers': CONFIG['num_layers'],
                'dropout': CONFIG['dropout'],
                'model_type': 'FastSeq2Seq'
            }
        }, model_path, pickle_protocol=4)
        
        scaler_path = f"{MODELS_FOLDER}/{scenario}_seq2seq_3050_scaler.pkl"
        joblib.dump(result['scaler'], scaler_path)
        print(f"  ✓ Saved: {model_path}")
        
        # Plot
        plot_results(scenario, result, CONFIG, f"{VISUALS_FOLDER}/{scenario}_seq2seq_3050.png")
        
        all_results[scenario] = {
            'mse': mse, 'mae': mae,
            'epochs': result['epochs'],
            'time': result['time_min']
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n\n" + "="*80)
print("🏁 TRAINING COMPLETE")
print("="*80)

if all_results:
    print(f"\n{'Scenario':<12} {'MSE':<10} {'MAE':<10} {'Epochs':<8} {'Time':<8}")
    print("-"*50)
    for s, r in all_results.items():
        status = "🏆" if r['mse'] < 0.1 else "✨" if r['mse'] < 0.3 else "⚠️"
        print(f"{s:<12} {r['mse']:<10.6f} {r['mae']:<10.6f} {r['epochs']:<8} {r['time']:<8.1f} {status}")

print(f"\n⏱️ Total: {(time.time()-total_start)/60:.1f} min")
print(f"📁 Models: {MODELS_FOLDER}/*_seq2seq_3050.pt")
print("="*80)

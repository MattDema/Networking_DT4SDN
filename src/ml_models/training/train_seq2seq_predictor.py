# train_seq2seq_predictor.py
"""
SEQUENCE-TO-SEQUENCE TRAFFIC PREDICTOR - RTX 3050 (4GB) OPTIMIZED

Same style as classifier but predicts actual values for graph visualization:
- Input: 90 seconds of history
- Output: 60 future traffic values (one per second)
- Used alongside classifier for realistic graph predictions

Architecture: Encoder-Decoder BiLSTM with Attention
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
import joblib
import warnings
import gc
warnings.filterwarnings('ignore')

# Memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("="*80)
print("🖥️ SEQ2SEQ TRAFFIC PREDICTOR - RTX 3050 OPTIMIZED 🖥️")
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
    print("✓ cuDNN benchmark enabled")
else:
    device = torch.device('cpu')
    print("⚠️ Running on CPU")

# =============================================================================
# RTX 3050 OPTIMIZED CONFIG
# =============================================================================
CONFIG = {
    # Data
    'sequence_length': 90,        # 1.5 minutes history
    'prediction_horizon': 60,     # Predict 60 future values
    
    # Encoder-Decoder architecture (memory optimized)
    'encoder_hidden': 128,
    'decoder_hidden': 128,
    'num_layers': 2,
    'dropout': 0.3,
    
    # Batch size optimized for 4GB
    'batch_size': 64,
    'accumulation_steps': 4,      # Effective batch: 256
    
    # Training
    'epochs': 500,
    'learning_rate': 0.001,
    'patience': 50,
    'weight_decay': 1e-4,
    
    # Data processing
    'smooth_window': 5,
    'teacher_forcing_ratio': 0.5,
}

SCENARIOS = {
    #'normal': 'network_data_normal_50000_*.csv',
    #'burst': 'network_data_burst_50000_*.csv', 
    #'congestion': 'network_data_congestion_50000_*.csv',
    #'ddos': 'network_data_ddos_50000_*.csv',
    'mixed': 'network_data_mixed_50000_*.csv'
}

DATA_FOLDER = 'src/ml_models/data_collection'
MODELS_FOLDER = 'models'
VISUALS_FOLDER = 'visuals/training'
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(VISUALS_FOLDER, exist_ok=True)

print("\n" + "="*80)
print("⚡ SEQ2SEQ CONFIGURATION ⚡")
print("="*80)
print(f"  📊 Input Sequence:      {CONFIG['sequence_length']}s history")
print(f"  🎯 Output Sequence:     {CONFIG['prediction_horizon']} future values")
print(f"  🧠 Encoder Hidden:      {CONFIG['encoder_hidden']} (BiLSTM)")
print(f"  🔮 Decoder Hidden:      {CONFIG['decoder_hidden']} (LSTM + Attention)")
print(f"  📚 Layers:              {CONFIG['num_layers']}")
print(f"  📦 Batch Size:          {CONFIG['batch_size']} (effective: {CONFIG['batch_size'] * CONFIG['accumulation_steps']})")
print(f"  ⏱️  Max Epochs:          {CONFIG['epochs']}")
print("="*80)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class Encoder(nn.Module):
    """Bidirectional LSTM Encoder"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        # Combine bidirectional states
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
        cell = cell.view(self.num_layers, 2, -1, self.hidden_size)
        cell = cell[:, 0, :, :] + cell[:, 1, :, :]
        return outputs, (hidden, cell)


class Attention(nn.Module):
    """Attention mechanism for decoder"""
    
    def __init__(self, encoder_hidden, decoder_hidden):
        super().__init__()
        self.attn = nn.Linear(encoder_hidden * 2 + decoder_hidden, decoder_hidden)
        self.v = nn.Linear(decoder_hidden, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    """LSTM Decoder with Attention"""
    
    def __init__(self, hidden_size, num_layers, dropout, encoder_hidden):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.attention = Attention(encoder_hidden, hidden_size)
        self.lstm = nn.LSTM(
            input_size=1 + encoder_hidden * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size + encoder_hidden * 2 + 1, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        # Ensure input is (batch, 1, 1) for concatenation
        if input.dim() == 1:
            input = input.unsqueeze(1).unsqueeze(2)  # (batch,) -> (batch, 1, 1)
        elif input.dim() == 2:
            input = input.unsqueeze(2)  # (batch, 1) -> (batch, 1, 1)
        
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden*2)
        
        lstm_input = torch.cat([input, context], dim=2)  # (batch, 1, 1 + hidden*2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        output = torch.cat([output.squeeze(1), context.squeeze(1), input.squeeze(1).squeeze(1).unsqueeze(1)], dim=1)
        prediction = self.fc(self.dropout(output))
        
        return prediction, hidden, cell


class Seq2SeqPredictor(nn.Module):
    """Full Sequence-to-Sequence Traffic Predictor"""
    
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            input_size=1,
            hidden_size=config['encoder_hidden'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        self.decoder = Decoder(
            hidden_size=config['decoder_hidden'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            encoder_hidden=config['encoder_hidden']
        )
        self.prediction_horizon = config['prediction_horizon']
        
    def forward(self, src, target=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        outputs = []
        input = src[:, -1, 0]
        
        for t in range(self.prediction_horizon):
            prediction, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs.append(prediction)
            
            if target is not None and np.random.random() < teacher_forcing_ratio:
                input = target[:, t]
            else:
                input = prediction.squeeze(1)
        
        return torch.cat(outputs, dim=1)


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
def load_and_prepare_data(scenario, pattern, data_folder, config):
    """Load data and create sequences"""
    
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
    
    # Smooth
    smoothed = pd.Series(raw_data).rolling(
        window=config['smooth_window'], 
        center=True,
        min_periods=1
    ).mean().values
    
    # CRITICAL: Clip negative values (data artifacts)
    smoothed = np.clip(smoothed, 0, None)
    
    # Stats
    print(f"  📊 Data stats:")
    print(f"     Min:  {np.min(smoothed):,.0f}")
    print(f"     Mean: {np.mean(smoothed):,.0f}")
    print(f"     Max:  {np.max(smoothed):,.0f}")
    
    return smoothed.reshape(-1, 1)


def create_sequences(data, seq_len, pred_horizon):
    """Create input-output sequences"""
    X, y = [], []
    
    for i in range(len(data) - seq_len - pred_horizon):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_horizon].flatten())
    
    return np.array(X), np.array(y)


# =============================================================================
# TRAINING
# =============================================================================
def train_seq2seq(scenario, data, config):
    """Train seq2seq model"""
    
    seq_len = config['sequence_length']
    pred_horizon = config['prediction_horizon']
    
    # Normalize
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_len, pred_horizon)
    print(f"  ✓ Created {len(X):,} sequences")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    
    val_size = int(len(X_train) * 0.15)
    indices = np.random.permutation(len(X_train))
    X_val, y_val = X_train[indices[:val_size]], y_train[indices[:val_size]]
    X_train, y_train = X_train[indices[val_size:]], y_train[indices[val_size:]]
    
    print(f"  📊 Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # DataLoaders
    train_dataset = Seq2SeqDataset(X_train, y_train)
    val_dataset = Seq2SeqDataset(X_val, y_val)
    test_dataset = Seq2SeqDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=0, pin_memory=True)
    
    # Model
    model = Seq2SeqPredictor(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model params: {total_params:,}")
    
    if torch.cuda.is_available():
        print(f"  📦 GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                           weight_decay=config['weight_decay'])
    
    # Warmup + cosine scheduler
    def warmup_cosine_schedule(epoch, warmup=20, total=config['epochs']):
        if epoch < warmup:
            return epoch / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (total - warmup)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_schedule)
    
    # Loss
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
        
        optimizer.zero_grad()
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            output = model(X_batch, y_batch, teacher_forcing_ratio=config['teacher_forcing_ratio'])
            loss = criterion(output, y_batch) / config['accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % config['accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * config['accumulation_steps']
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch, y_batch, teacher_forcing_ratio=0)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
        
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
        
        # Progress
        if (epoch + 1) % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            gpu_mem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
            print(f"     Epoch {epoch+1:4d} | Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | LR: {lr:.2e} | GPU: {gpu_mem:.1f}GB")
        
        if patience_counter >= config['patience']:
            print(f"  ⏹️ Early stop at epoch {epoch+1}")
            break
    
    # Restore best
    model.load_state_dict(best_state)
    
    # Evaluate on test
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch, teacher_forcing_ratio=0)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # Cleanup
    del train_loader, val_loader, train_dataset, val_dataset
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'model': model,
        'scaler': scaler,
        'mse': mse,
        'mae': mae,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_trained': len(train_losses),
        'predictions': all_preds,
        'targets': all_targets
    }


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_seq2seq_results(scenario, result, config, save_path):
    """Plot seq2seq training results"""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'SEQ2SEQ PREDICTOR: {scenario.upper()} (RTX 3050)', 
                 fontsize=14, fontweight='bold')
    
    # 1. Training curves
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(result['train_losses'], 'b-', label='Train Loss', alpha=0.7)
    ax1.plot(result['val_losses'], 'r-', label='Val Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Sample predictions
    ax2 = plt.subplot(2, 2, 2)
    idx = np.random.randint(0, len(result['predictions']))
    ax2.plot(result['targets'][idx], 'b-', label='Actual', linewidth=2)
    ax2.plot(result['predictions'][idx], 'r--', label='Predicted', linewidth=2)
    ax2.set_xlabel('Future timestep')
    ax2.set_ylabel('Traffic (scaled)')
    ax2.set_title('Sample Prediction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Prediction error distribution
    ax3 = plt.subplot(2, 2, 3)
    errors = (result['predictions'] - result['targets']).flatten()
    ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--')
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Count')
    ax3.set_title('Error Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    mse = result['mse']
    mae = result['mae']
    
    status = '🏆 EXCELLENT' if mse < 0.1 else '✨ GOOD' if mse < 0.3 else '⚠️ OK'
    color = '#27ae60' if mse < 0.1 else '#f39c12' if mse < 0.3 else '#e74c3c'
    
    text = f"""
    ════════════════════════════════════
       SEQ2SEQ RESULTS
    ════════════════════════════════════
    
    📊 MSE:           {mse:.6f}
    📈 MAE:           {mae:.6f}
    
    ⏱️  Epochs:        {result['epochs_trained']}
    🔮 Predicts:      {config['prediction_horizon']} future values
    📜 History:       {config['sequence_length']}s
    
    ════════════════════════════════════
    STATUS: {status}
    ════════════════════════════════════
    """
    
    ax4.text(0.5, 0.5, text, transform=ax4.transAxes, fontsize=11,
             family='monospace', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot saved: {save_path}")


def plot_comparison(results, save_path):
    """Compare all scenarios"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = list(results.keys())
    mses = [results[s]['mse'] for s in scenarios]
    maes = [results[s]['mae'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    colors = ['#27ae60' if m < 0.1 else '#f39c12' if m < 0.3 else '#e74c3c' for m in mses]
    
    bars1 = ax.bar(x - width/2, mses, width, label='MSE', color=colors, edgecolor='black')
    bars2 = ax.bar(x + width/2, maes, width, label='MAE', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Error')
    ax.set_title('SEQ2SEQ PREDICTOR - All Scenarios (RTX 3050)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in scenarios])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Comparison saved: {save_path}")


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
all_results = {}
total_start = time.time()

for idx, (scenario, pattern) in enumerate(SCENARIOS.items(), 1):
    print(f"\n\n{'='*80}")
    print(f"[{idx}/{len(SCENARIOS)}] TRAINING SEQ2SEQ: {scenario.upper()}")
    print('='*80)
    
    try:
        # Cleanup before each scenario
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
            print(f"  🧹 GPU cleared. Available: {free_mem:.1f} GB")
        
        # Load data
        data = load_and_prepare_data(scenario, pattern, DATA_FOLDER, CONFIG)
        
        # Train
        scenario_start = time.time()
        result = train_seq2seq(scenario, data, CONFIG)
        training_time = time.time() - scenario_start
        
        print(f"\n  📊 FINAL RESULTS:")
        print(f"     MSE: {result['mse']:.6f}")
        print(f"     MAE: {result['mae']:.6f}")
        print(f"     Time: {training_time/60:.1f} min")
        
        if result['mse'] < 0.1:
            print(f"     Status: 🏆 EXCELLENT!")
        elif result['mse'] < 0.3:
            print(f"     Status: ✨ GOOD")
        else:
            print(f"     Status: ⚠️ Needs improvement")
        
        # Save model
        model_path = f"{MODELS_FOLDER}/{scenario}_seq2seq_3050.pt"
        scaler_path = f"{MODELS_FOLDER}/{scenario}_seq2seq_3050_scaler.pkl"
        
        torch.save({
            'model_state_dict': result['model'].state_dict(),
            'config': {
                'sequence_length': CONFIG['sequence_length'],
                'prediction_horizon': CONFIG['prediction_horizon'],
                'encoder_hidden': CONFIG['encoder_hidden'],
                'decoder_hidden': CONFIG['decoder_hidden'],
                'num_layers': CONFIG['num_layers'],
                'dropout': CONFIG['dropout'],
                'model_type': 'Seq2SeqPredictor'
            }
        }, model_path, pickle_protocol=4)  # Protocol 4 for Python 3.8 compatibility
        joblib.dump(result['scaler'], scaler_path)
        print(f"  ✓ Saved: {model_path}")
        
        # Plot
        vis_path = f"{VISUALS_FOLDER}/{scenario}_seq2seq_3050.png"
        plot_seq2seq_results(scenario, result, CONFIG, vis_path)
        
        all_results[scenario] = {
            'mse': result['mse'],
            'mae': result['mae'],
            'training_time_min': training_time / 60,
            'epochs': result['epochs_trained']
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# Comparison plot
if len(all_results) > 1:
    comparison_path = f"{VISUALS_FOLDER}/all_seq2seq_3050_comparison.png"
    plot_comparison(all_results, comparison_path)

# Final summary
print("\n\n" + "="*80)
print("🏁 SEQ2SEQ TRAINING COMPLETE")
print("="*80)

if all_results:
    print(f"\n{'Scenario':<12} {'MSE':<12} {'MAE':<12} {'Time':<8} {'Epochs':<8} {'Status'}")
    print("-"*70)
    for s, m in all_results.items():
        status = "🏆" if m['mse'] < 0.1 else "✨" if m['mse'] < 0.3 else "⚠️"
        print(f"{s:<12} {m['mse']:<12.6f} {m['mae']:<12.6f} {m['training_time_min']:<8.1f} {m['epochs']:<8} {status}")
    
    avg_mse = np.mean([m['mse'] for m in all_results.values()])
    print("-"*70)
    print(f"{'Average:':<12} {avg_mse:.6f}")

print(f"\n⏱️ Total time: {(time.time()-total_start)/60:.1f} minutes")
print(f"\n📁 Models: {MODELS_FOLDER}/*_seq2seq_3050.pt")
print(f"📊 Plots: {VISUALS_FOLDER}/*_seq2seq_3050.png")

if torch.cuda.is_available():
    gpu_mem = torch.cuda.max_memory_allocated()/1e9
    print(f"💾 Peak GPU memory: {gpu_mem:.1f} GB")

print("="*80)

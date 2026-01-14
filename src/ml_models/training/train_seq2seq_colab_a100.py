# train_seq2seq_colab_a100.py
"""
SEQ2SEQ TRAFFIC PREDICTOR - COLAB A100 OPTIMIZED (40GB VRAM)

Fast training version for Colab Pro with A100 GPU.
Target: All 5 scenarios in ~30 minutes total.

Optimizations:
- Massive batch sizes (512)
- No gradient accumulation
- Fewer epochs with aggressive early stopping
- Mixed precision training (FP16)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
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
print("🚀 SEQ2SEQ - COLAB A100 FAST TRAINING 🚀")
print("="*80)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# GPU SETUP
# =============================================================================
assert torch.cuda.is_available(), "GPU required!"
device = torch.device('cuda')
gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"\n🎮 GPU: {gpu_name}")
print(f"💾 GPU Memory: {gpu_memory:.1f} GB")

# Enable TF32 for A100 speedup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
print("✓ TF32 and cuDNN benchmark enabled")

# =============================================================================
# A100 OPTIMIZED CONFIG - MAXIMUM SPEED
# =============================================================================
CONFIG = {
    # Data
    'sequence_length': 90,
    'prediction_horizon': 60,
    
    # Larger model (A100 can handle it)
    'encoder_hidden': 256,
    'decoder_hidden': 256,
    'num_layers': 2,
    'dropout': 0.2,
    
    # MASSIVE batch for A100
    'batch_size': 512,
    'accumulation_steps': 1,  # No need for accumulation
    
    # Fewer epochs, aggressive stopping
    'epochs': 200,
    'learning_rate': 0.002,  # Higher LR for faster convergence
    'patience': 20,          # Stop faster
    'weight_decay': 1e-4,
    
    # Data processing
    'smooth_window': 5,
    'teacher_forcing_ratio': 0.5,
    
    # Mixed precision
    'use_amp': True,
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
print("⚡ A100 OPTIMIZED CONFIG ⚡")
print("="*80)
print(f"  📊 Sequence:    {CONFIG['sequence_length']}s → {CONFIG['prediction_horizon']} values")
print(f"  🧠 Hidden:      {CONFIG['encoder_hidden']} units")
print(f"  📦 Batch:       {CONFIG['batch_size']} (MASSIVE)")
print(f"  ⏱️  Epochs:      {CONFIG['epochs']} (early stop @ {CONFIG['patience']})")
print(f"  🔧 AMP:         {'ENABLED' if CONFIG['use_amp'] else 'disabled'}")
print("="*80)

# =============================================================================
# MODEL (same architecture)
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
        cell = cell.view(self.num_layers, 2, -1, self.hidden_size)
        cell = cell[:, 0, :, :] + cell[:, 1, :, :]
        return outputs, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, encoder_hidden, decoder_hidden):
        super().__init__()
        self.attn = nn.Linear(encoder_hidden * 2 + decoder_hidden, decoder_hidden)
        self.v = nn.Linear(decoder_hidden, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, encoder_hidden):
        super().__init__()
        self.attention = Attention(encoder_hidden, hidden_size)
        self.lstm = nn.LSTM(1 + encoder_hidden * 2, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size + encoder_hidden * 2 + 1, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        if input.dim() == 1:
            input = input.unsqueeze(1).unsqueeze(2)
        elif input.dim() == 2:
            input = input.unsqueeze(2)
        
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        lstm_input = torch.cat([input, context], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        output = torch.cat([output.squeeze(1), context.squeeze(1), 
                           input.squeeze(1).squeeze(1).unsqueeze(1)], dim=1)
        return self.fc(self.dropout(output)), hidden, cell

class Seq2SeqPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(1, config['encoder_hidden'], config['num_layers'], config['dropout'])
        self.decoder = Decoder(config['decoder_hidden'], config['num_layers'], 
                              config['dropout'], config['encoder_hidden'])
        self.prediction_horizon = config['prediction_horizon']
        
    def forward(self, src, target=None, teacher_forcing_ratio=0.5):
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        outputs = []
        input = src[:, -1, 0]
        
        for t in range(self.prediction_horizon):
            prediction, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs.append(prediction)
            
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
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
def load_data(scenario, pattern, data_folder, config):
    full_pattern = os.path.join(data_folder, pattern)
    files = sorted(glob.glob(full_pattern))
    
    if not files:
        raise FileNotFoundError(f"No files: {full_pattern}")
    
    print(f"  📂 Loading {len(files)} files...")
    dfs = [pd.read_csv(f) for f in files]
    data = pd.concat(dfs, ignore_index=True)['bytes'].values.astype(np.float32)
    
    # Smooth and clip
    smoothed = pd.Series(data).rolling(window=config['smooth_window'], center=True, min_periods=1).mean().values
    smoothed = np.clip(smoothed, 0, None)
    
    print(f"  ✓ Samples: {len(smoothed):,}")
    return smoothed.reshape(-1, 1)

def create_sequences(data, seq_len, pred_horizon):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_horizon):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_horizon].flatten())
    return np.array(X), np.array(y)

# =============================================================================
# TRAINING
# =============================================================================
def train(scenario, data, config):
    seq_len = config['sequence_length']
    pred_horizon = config['prediction_horizon']
    
    # Normalize
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_len, pred_horizon)
    print(f"  ✓ Sequences: {len(X):,}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    val_size = int(len(X_train) * 0.15)
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train, y_train = X_train[val_size:], y_train[val_size:]
    
    print(f"  📊 Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # DataLoaders (more workers for A100)
    train_loader = DataLoader(Seq2SeqDataset(X_train, y_train), 
                             batch_size=config['batch_size'], shuffle=True, 
                             num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Seq2SeqDataset(X_val, y_val), 
                           batch_size=config['batch_size'], num_workers=2, pin_memory=True)
    test_loader = DataLoader(Seq2SeqDataset(X_test, y_test), 
                            batch_size=config['batch_size'], num_workers=2, pin_memory=True)
    
    # Model
    model = Seq2SeqPredictor(config).to(device)
    print(f"  ✓ Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.MSELoss()
    scaler_amp = GradScaler(enabled=config['use_amp'])
    
    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    best_state = None
    
    print(f"\n  🏋️ Training...")
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            with autocast(enabled=config['use_amp']):
                output = model(X_batch, y_batch, config['teacher_forcing_ratio'])
                loss = criterion(output, y_batch)
            
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with autocast(enabled=config['use_amp']):
                    output = model(X_batch, teacher_forcing_ratio=0)
                    val_loss += criterion(output, y_batch).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"     Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if patience_counter >= config['patience']:
            print(f"  ⏹️ Early stop @ {epoch+1}")
            break
    
    # Restore best
    model.load_state_dict(best_state)
    model.to(device)
    
    # Test
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch.to(device), teacher_forcing_ratio=0)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mse = np.mean((all_preds - all_targets) ** 2)
    
    # Cleanup
    del train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return {'model': model, 'scaler': scaler, 'mse': mse, 
            'train_losses': train_losses, 'val_losses': val_losses,
            'epochs': len(train_losses)}

# =============================================================================
# MAIN
# =============================================================================
all_results = {}
total_start = time.time()

for idx, (scenario, pattern) in enumerate(SCENARIOS.items(), 1):
    print(f"\n{'='*60}")
    print(f"[{idx}/{len(SCENARIOS)}] {scenario.upper()}")
    print('='*60)
    
    try:
        torch.cuda.empty_cache()
        gc.collect()
        
        data = load_data(scenario, pattern, DATA_FOLDER, CONFIG)
        start = time.time()
        result = train(scenario, data, CONFIG)
        elapsed = time.time() - start
        
        print(f"\n  ✓ MSE: {result['mse']:.6f}")
        print(f"  ✓ Time: {elapsed/60:.1f} min")
        
        # Save (compatible with VM)
        model_path = f"{MODELS_FOLDER}/{scenario}_seq2seq_3050.pt"
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
        }, model_path, pickle_protocol=4)
        
        scaler_path = f"{MODELS_FOLDER}/{scenario}_seq2seq_3050_scaler.pkl"
        joblib.dump(result['scaler'], scaler_path)
        print(f"  ✓ Saved: {model_path}")
        
        all_results[scenario] = {'mse': result['mse'], 'epochs': result['epochs'], 'time': elapsed/60}
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n\n" + "="*80)
print("🏁 TRAINING COMPLETE")
print("="*80)
total_time = (time.time() - total_start) / 60
for s, m in all_results.items():
    status = "🏆" if m['mse'] < 0.1 else "✨" if m['mse'] < 0.3 else "⚠️"
    print(f"{s:<12} MSE: {m['mse']:.6f}  Epochs: {m['epochs']:<4}  Time: {m['time']:.1f}min  {status}")
print("-"*60)
print(f"TOTAL TIME: {total_time:.1f} minutes")
print(f"Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print("="*80)

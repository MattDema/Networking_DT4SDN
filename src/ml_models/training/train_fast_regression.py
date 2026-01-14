# train_fast_regression.py
"""
ULTRA FAST REGRESSION - Direct prediction (no autoregressive loop)

Instead of seq2seq (slow 60-step loop), this predicts all 60 values AT ONCE.
Should train in ~2-3 minutes per scenario!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import gc
import time

print("="*60)
print("ULTRA FAST REGRESSION (No autoregressive loop)")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

CONFIG = {
    'sequence_length': 90,
    'prediction_horizon': 60,
    'hidden': 256,
    'batch_size': 256,
    'epochs': 100,
    'lr': 0.002,
    'patience': 10,
}

# Simple fast model - predicts ALL 60 values at once
class FastPredictor(nn.Module):
    def __init__(self, seq_len, pred_len, hidden):
        super().__init__()
        # 1D CNN for feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # BiLSTM for temporal patterns
        self.lstm = nn.LSTM(128, hidden, 2, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Direct output - ALL 60 values at once!
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, pred_len)  # Direct to 60 outputs
        )
        
    def forward(self, x):
        # x: (batch, seq, 1)
        x = x.permute(0, 2, 1)  # (batch, 1, seq)
        x = self.conv(x)        # (batch, 128, seq/4)
        x = x.permute(0, 2, 1)  # (batch, seq/4, 128)
        
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)  # (batch, hidden*2)
        
        return self.fc(h)  # (batch, 60) - ALL AT ONCE!

SCENARIOS = {
    'mixed': 'network_data_mixed_50000_*.csv',
    'normal': 'network_data_normal_50000_*.csv',
    'burst': 'network_data_burst_50000_*.csv',
    'congestion': 'network_data_congestion_50000_*.csv',
    'ddos': 'network_data_ddos_50000_*.csv',
}

DATA_FOLDER = 'src/ml_models/data_collection'
os.makedirs('models', exist_ok=True)

class DS(Dataset):
    def __init__(self, x, y): 
        self.x, self.y = torch.FloatTensor(x), torch.FloatTensor(y)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

all_results = {}
total_start = time.time()

for scenario, pattern in SCENARIOS.items():
    print(f"\n{'='*40}")
    print(f"Training: {scenario.upper()}")
    print('='*40)
    
    start = time.time()
    
    try:
        # Load
        files = glob.glob(os.path.join(DATA_FOLDER, pattern))
        if not files:
            print(f"  No files found, skipping")
            continue
            
        data = pd.concat([pd.read_csv(f) for f in files])['bytes'].values.astype(np.float32)
        data = pd.Series(data).rolling(5, center=True, min_periods=1).mean().values
        data = np.clip(data, 0, None).reshape(-1, 1)
        print(f"  Samples: {len(data):,}")
        
        # Normalize
        scaler = RobustScaler()
        data = scaler.fit_transform(data)
        
        # Sequences
        X, y = [], []
        for i in range(len(data) - 150):
            X.append(data[i:i+90])
            y.append(data[i+90:i+150].flatten())
        X, y = np.array(X), np.array(y)
        
        # Split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42)
        
        # Loaders
        train_dl = DataLoader(DS(X_tr, y_tr), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=0)
        val_dl = DataLoader(DS(X_va, y_va), batch_size=CONFIG['batch_size'], num_workers=0)
        
        # Model
        model = FastPredictor(90, 60, CONFIG['hidden']).to(device)
        opt = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
        crit = nn.MSELoss()
        
        best_loss = float('inf')
        patience = 0
        best_state = None
        
        for ep in range(CONFIG['epochs']):
            # Train
            model.train()
            tr_loss = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                tr_loss += loss.item()
            tr_loss /= len(train_dl)
            
            # Val
            model.eval()
            va_loss = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    va_loss += crit(model(xb), yb).item()
            va_loss /= len(val_dl)
            
            if va_loss < best_loss:
                best_loss = va_loss
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            
            if (ep+1) % 10 == 0:
                print(f"  Ep {ep+1}: train={tr_loss:.4f} val={va_loss:.4f}")
            
            if patience >= CONFIG['patience']:
                print(f"  Early stop @ {ep+1}")
                break
        
        elapsed = time.time() - start
        
        # Save
        torch.save({
            'model_state_dict': best_state,
            'config': CONFIG,
            'model_type': 'FastPredictor'
        }, f'models/{scenario}_seq2seq_3050.pt', pickle_protocol=4)
        joblib.dump(scaler, f'models/{scenario}_seq2seq_3050_scaler.pkl')
        
        all_results[scenario] = {'mse': best_loss, 'time': elapsed}
        print(f"  ✓ MSE: {best_loss:.6f} | Time: {elapsed:.0f}s")
        
        # Cleanup
        del model, train_dl, val_dl
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
total_time = time.time() - total_start
for s, r in all_results.items():
    print(f"{s:<12} MSE: {r['mse']:.6f}  Time: {r['time']:.0f}s")
print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f} min)")

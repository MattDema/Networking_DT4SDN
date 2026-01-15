# train_state_classifier_laptop.py
"""
TRAFFIC STATE CLASSIFIER - RTX 3050 (4GB) OPTIMIZED VERSION

Same features as the A100 version but optimized for 4GB VRAM:
- Pure LSTM (no Transformer to save memory)
- Optimized batch/sequence sizes
- Full visualizations and reporting
- Predicts: NORMAL, ELEVATED, HIGH, CRITICAL

Output: Probability of each state 60s ahead → trigger SDN mitigation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import warnings
import gc
warnings.filterwarnings('ignore')

# Memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("="*80)
print("🖥️ TRAFFIC STATE CLASSIFIER - RTX 3050 OPTIMIZED 🖥️")
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
# RTX 3050 OPTIMIZED CONFIG - Maximum for 4GB VRAM
# =============================================================================
CONFIG = {
    # Dynamic thresholds
    'use_dynamic_thresholds': True,
    'percentiles': {
        'NORMAL': 25,
        'ELEVATED': 50,
        'HIGH': 75,
    },
    'class_names': ['NORMAL', 'ELEVATED', 'HIGH', 'CRITICAL'],
    
    # OPTIMIZED for 4GB - Maximum possible
    'sequence_length': 90,        # 1.5 minutes history
    'prediction_horizon': 60,     # Predict 60s ahead
    
    # LSTM-only architecture (no Transformer to save VRAM)
    'lstm_hidden': 128,           # Balanced size
    'lstm_layers': 2,             # 2 BiLSTM layers
    'dropout': 0.3,
    
    # Batch size optimized for 4GB
    'batch_size': 128,            # Max stable for 4GB
    'accumulation_steps': 4,      # Effective batch: 512
    
    # Training - aggressive!
    'epochs': 1000,               # Many epochs
    'learning_rate': 0.001,
    'patience': 100,              # Generous patience
    'weight_decay': 1e-4,
    
    # Single model (no ensemble to save time)
    'num_ensemble': 1,
    
    # Data augmentation
    'augment': True,
    'noise_std': 0.05,
    
    # Smoothing
    'smooth_window': 10,
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
print("⚡ RTX 3050 OPTIMIZED CONFIGURATION ⚡")
print("="*80)
print(f"  📊 Sequence Length:     {CONFIG['sequence_length']}s")
print(f"  🎯 Prediction Horizon:  {CONFIG['prediction_horizon']}s ahead")
print(f"  🧠 LSTM Hidden:         {CONFIG['lstm_hidden']} (BiLSTM)")
print(f"  📚 LSTM Layers:         {CONFIG['lstm_layers']}")
print(f"  📦 Batch Size:          {CONFIG['batch_size']} (effective: {CONFIG['batch_size'] * CONFIG['accumulation_steps']})")
print(f"  ⏱️  Max Epochs:          {CONFIG['epochs']}")
print(f"  🏷️  Classes:             {CONFIG['class_names']}")
print("="*80)

# =============================================================================
# BILSTM CLASSIFIER (Memory efficient - no Transformer)
# =============================================================================
class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM Classifier - optimized for low VRAM
    """
    
    def __init__(self, input_size, lstm_hidden, lstm_layers, num_classes, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(lstm_hidden // 2, num_classes)
        )
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
        
        # Classify
        return self.classifier(context)

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
def bytes_to_state(value, thresholds):
    """Convert byte value to traffic state using dynamic thresholds"""
    if value < thresholds['NORMAL']:
        return 0
    elif value < thresholds['ELEVATED']:
        return 1
    elif value < thresholds['HIGH']:
        return 2
    return 3

def calculate_dynamic_thresholds(data, percentiles):
    """Calculate thresholds from actual data distribution"""
    return {
        'NORMAL': np.percentile(data, percentiles['NORMAL']),
        'ELEVATED': np.percentile(data, percentiles['ELEVATED']),
        'HIGH': np.percentile(data, percentiles['HIGH']),
    }

def load_and_prepare_classification_data(scenario, pattern, data_folder, config):
    """Load data and convert to classification labels with DYNAMIC thresholds"""
    
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
    
    # Dynamic thresholds
    thresholds = calculate_dynamic_thresholds(smoothed, config['percentiles'])
    print(f"  🎯 Dynamic thresholds:")
    print(f"     NORMAL:   < {thresholds['NORMAL']:,.0f} bytes")
    print(f"     ELEVATED: < {thresholds['ELEVATED']:,.0f} bytes")
    print(f"     HIGH:     < {thresholds['HIGH']:,.0f} bytes")
    print(f"     CRITICAL: >= {thresholds['HIGH']:,.0f} bytes")
    
    # Convert to states
    states = np.array([bytes_to_state(v, thresholds) for v in smoothed])
    
    # Class distribution
    unique, counts = np.unique(states, return_counts=True)
    print(f"  📊 Class distribution:")
    for u, c in zip(unique, counts):
        pct = c / len(states) * 100
        print(f"     {config['class_names'][u]}: {c:,} ({pct:.1f}%)")
    
    return raw_data.reshape(-1, 1), states, thresholds

def create_classification_sequences(data, states, seq_len, pred_horizon):
    """Create sequences with future state as target"""
    X, y = [], []
    
    for i in range(len(data) - seq_len - pred_horizon):
        X.append(data[i:i + seq_len])
        y.append(states[i + seq_len + pred_horizon - 1])
    
    return np.array(X), np.array(y)

class TrafficDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.05):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.noise_std = noise_std
        self._training = True
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and self._training:
            noise = torch.randn_like(x) * self.noise_std * x.std()
            x = x + noise
        return x, self.y[idx]
    
    def train(self):
        self._training = True
    
    def eval(self):
        self._training = False

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_classifier(scenario, data, states, config, model_idx=0):
    """Train a BiLSTM classifier"""
    
    seq_len = config['sequence_length']
    pred_horizon = config['prediction_horizon']
    
    # Normalize
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_classification_sequences(data_scaled, states, seq_len, pred_horizon)
    print(f"  ✓ Created {len(X):,} sequences")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42 + model_idx
    )
    
    val_size = int(len(X_train) * 0.15)
    indices = np.random.permutation(len(X_train))
    X_val, y_val = X_train[indices[:val_size]], y_train[indices[:val_size]]
    X_train, y_train = X_train[indices[val_size:]], y_train[indices[val_size:]]
    
    print(f"  📊 Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # Class weights
    class_counts = np.bincount(y_train, minlength=4)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"  ⚖️ Class weights: {class_weights.cpu().numpy().round(2)}")
    
    # DataLoaders
    train_dataset = TrafficDataset(X_train, y_train, augment=config['augment'], noise_std=config['noise_std'])
    val_dataset = TrafficDataset(X_val, y_val)
    test_dataset = TrafficDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=0, pin_memory=True)
    
    # Model
    model = BiLSTMClassifier(
        input_size=1,
        lstm_hidden=config['lstm_hidden'],
        lstm_layers=config['lstm_layers'],
        num_classes=len(config['class_names']),
        dropout=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model params: {total_params:,}")
    
    if torch.cuda.is_available():
        print(f"  📦 GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                           weight_decay=config['weight_decay'])
    
    # Cosine scheduler with warmup
    def warmup_cosine_schedule(epoch, warmup=20, total=config['epochs']):
        if epoch < warmup:
            return epoch / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (total - warmup)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_schedule)
    
    # Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_state = None
    
    print(f"\n  🏋️ Training (max {config['epochs']} epochs)...")
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_dataset.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        optimizer.zero_grad()
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch) / config['accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % config['accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * config['accumulation_steps']
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        model.eval()
        val_dataset.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Progress
        if (epoch + 1) % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            gpu_mem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
            print(f"     Epoch {epoch+1:4d} | Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
                  f"LR: {lr:.2e} | GPU: {gpu_mem:.1f}GB")
        
        if patience_counter >= config['patience']:
            print(f"  ⏹️ Early stop at epoch {epoch+1}")
            break
    
    # Restore best
    model.load_state_dict(best_state)
    
    # Evaluate on test
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # Cleanup
    del train_loader, val_loader, train_dataset, val_dataset
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'f1_score': f1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'epochs_trained': len(train_losses),
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }

# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_classifier_results(scenario, results, config, save_path):
    """Plot comprehensive classification results"""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'STATE CLASSIFIER: {scenario.upper()} (RTX 3050)', 
                 fontsize=14, fontweight='bold')
    
    result = results[0]
    
    # 1. Training curves
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(result['train_accs'], 'b-', label='Train Acc', alpha=0.7)
    ax1.plot(result['val_accs'], 'r-', label='Val Acc', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Confusion matrix
    ax2 = plt.subplot(2, 2, 2)
    cm = confusion_matrix(result['targets'], result['predictions'])
    im = ax2.imshow(cm, cmap='Blues')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(config['class_names'], rotation=45)
    ax2.set_yticklabels(config['class_names'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix')
    
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, cm[i, j], ha='center', va='center', fontsize=10,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.colorbar(im, ax=ax2)
    
    # 3. Per-class accuracy
    ax3 = plt.subplot(2, 2, 3)
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
    colors = ['#27ae60' if a > 0.7 else '#f39c12' if a > 0.5 else '#e74c3c' for a in per_class_acc]
    bars = ax3.bar(config['class_names'], per_class_acc, color=colors, edgecolor='black')
    ax3.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good')
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='OK')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Per-Class Accuracy')
    ax3.set_ylim(0, 1)
    ax3.legend()
    for bar, val in zip(bars, per_class_acc):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=10)
    
    # 4. Summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    acc = result['accuracy']
    f1 = result['f1_score']
    
    status = '🏆 EXCELLENT' if acc > 0.8 else '✨ GOOD' if acc > 0.6 else '⚠️ OK'
    color = '#27ae60' if acc > 0.8 else '#f39c12' if acc > 0.6 else '#e74c3c'
    
    text = f"""
    ════════════════════════════════════
       CLASSIFICATION RESULTS
    ════════════════════════════════════
    
    📊 Accuracy:      {acc:.4f} ({acc*100:.1f}%)
    📈 F1 Score:      {f1:.4f}
    
    ⏱️  Epochs:        {result['epochs_trained']}
    🔮 Predicts:      {config['prediction_horizon']}s ahead
    📜 History:       {config['sequence_length']}s
    
    🏷️  Classes:       {', '.join(config['class_names'])}
    
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
    accuracies = [results[s]['accuracy'] for s in scenarios]
    f1_scores = [results[s]['f1_score'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    colors_acc = ['#27ae60' if a > 0.7 else '#f39c12' if a > 0.5 else '#e74c3c' for a in accuracies]
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color=colors_acc, edgecolor='black')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.7, edgecolor='black')
    
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    
    ax.set_ylabel('Score')
    ax.set_title('STATE CLASSIFIER - All Scenarios (RTX 3050)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in scenarios])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    
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
    print(f"[{idx}/{len(SCENARIOS)}] TRAINING CLASSIFIER: {scenario.upper()}")
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
        data, states, thresholds = load_and_prepare_classification_data(scenario, pattern, DATA_FOLDER, CONFIG)
        
        # Train
        scenario_results = []
        scenario_start = time.time()
        
        for m_idx in range(CONFIG['num_ensemble']):
            if CONFIG['num_ensemble'] > 1:
                print(f"\n  --- Model {m_idx+1}/{CONFIG['num_ensemble']} ---")
            result = train_classifier(scenario, data, states, CONFIG, model_idx=m_idx)
            scenario_results.append(result)
            print(f"\n  ✓ Accuracy: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}")
        
        training_time = time.time() - scenario_start
        
        # Results
        acc = scenario_results[0]['accuracy']
        f1 = scenario_results[0]['f1_score']
        
        print(f"\n  📊 FINAL RESULTS:")
        print(f"     Accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print(f"     F1 Score: {f1:.4f}")
        print(f"     Time:     {training_time/60:.1f} min")
        
        if acc > 0.8:
            print(f"     Status:   🏆 EXCELLENT!")
        elif acc > 0.6:
            print(f"     Status:   ✨ GOOD")
        else:
            print(f"     Status:   ⚠️ Needs improvement")
        
        # Save model
        model_path = f"{MODELS_FOLDER}/{scenario}_classifier_3050.pt"
        scaler_path = f"{MODELS_FOLDER}/{scenario}_classifier_3050_scaler.pkl"
        
        torch.save({
            'model_state_dict': scenario_results[0]['model'].state_dict(),
            'config': {
                'sequence_length': CONFIG['sequence_length'],
                'prediction_horizon': CONFIG['prediction_horizon'],
                'class_names': CONFIG['class_names'],
                'thresholds': thresholds,
                'lstm_hidden': CONFIG['lstm_hidden'],
                'lstm_layers': CONFIG['lstm_layers'],
                'model_type': 'BiLSTMClassifier'
            }
        }, model_path)
        joblib.dump(scenario_results[0]['scaler'], scaler_path)
        print(f"  ✓ Saved: {model_path}")
        
        # Plot
        vis_path = f"{VISUALS_FOLDER}/{scenario}_classifier_3050.png"
        plot_classifier_results(scenario, scenario_results, CONFIG, vis_path)
        
        all_results[scenario] = {
            'accuracy': acc,
            'f1_score': f1,
            'training_time_min': training_time / 60,
            'epochs': scenario_results[0]['epochs_trained']
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# Comparison plot
if len(all_results) > 1:
    comparison_path = f"{VISUALS_FOLDER}/all_classifier_3050_comparison.png"
    plot_comparison(all_results, comparison_path)

# Final summary
print("\n\n" + "="*80)
print("🏁 CLASSIFICATION TRAINING COMPLETE")
print("="*80)

if all_results:
    print(f"\n{'Scenario':<12} {'Accuracy':<10} {'F1':<10} {'Time':<8} {'Epochs':<8} {'Status'}")
    print("-"*70)
    for s, m in all_results.items():
        status = "🏆" if m['accuracy'] > 0.8 else "✨" if m['accuracy'] > 0.6 else "⚠️"
        print(f"{s:<12} {m['accuracy']:<10.4f} {m['f1_score']:<10.4f} {m['training_time_min']:<8.1f} {m['epochs']:<8} {status}")
    
    avg_acc = np.mean([m['accuracy'] for m in all_results.values()])
    print("-"*70)
    print(f"{'Average:':<12} {avg_acc:.4f}")
    
    good_count = sum(1 for m in all_results.values() if m['accuracy'] > 0.6)
    print(f"\n🎯 Good models (Acc > 0.6): {good_count}/{len(all_results)}")
    print(f"🔮 Predicts state {CONFIG['prediction_horizon']}s ahead")

print(f"\n⏱️ Total time: {(time.time()-total_start)/60:.1f} minutes")
print(f"\n📁 Models: {MODELS_FOLDER}/*_classifier_3050.pt")
print(f"📊 Plots: {VISUALS_FOLDER}/*_classifier_3050.png")

if torch.cuda.is_available():
    gpu_mem = torch.cuda.max_memory_allocated()/1e9
    print(f"💾 Peak GPU memory: {gpu_mem:.1f} GB")

print("="*80)

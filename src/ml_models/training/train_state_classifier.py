# train_state_classifier.py
"""
TRAFFIC STATE CLASSIFIER - Predicts traffic categories for SDN mitigation

🔥 MAXIMUM RESOURCE UTILIZATION 🔥
- Uses full A100 80GB GPU capacity
- Massive batch sizes (8192+)
- Deep Transformer + LSTM hybrid
- Ensemble of 3 models for robust predictions
- Predicts: NORMAL, ELEVATED, HIGH, CRITICAL

Output: Probability of each state 90s ahead → trigger SDN mitigation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import sys
import os
import time
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import warnings
import gc
warnings.filterwarnings('ignore')

# Set environment variable for better memory management
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("="*80)
print("🔥 TRAFFIC STATE CLASSIFIER - MAXIMUM A100 POWER 🔥")
print("="*80)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# GPU SETUP - Maximize A100 usage
# =============================================================================
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n🎮 GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    
    # Enable all A100 optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    
    print("✓ TF32 enabled for maximum throughput")
    print("✓ cuDNN benchmark enabled")
else:
    device = torch.device('cpu')
    print("⚠️ Running on CPU")

# =============================================================================
# EXTREME CONFIGURATION - PUSH THE LIMITS
# =============================================================================
CONFIG = {
    # DYNAMIC THRESHOLDS - calculated per scenario from actual data!
    # Uses percentiles to adapt to any traffic pattern
    'use_dynamic_thresholds': True,
    'percentiles': {
        'NORMAL': 25,      # Below 25th percentile = Normal
        'ELEVATED': 50,    # 25-50th percentile = Elevated
        'HIGH': 75,        # 50-75th percentile = High
        # Above 75th percentile = Critical
    },
    'class_names': ['NORMAL', 'ELEVATED', 'HIGH', 'CRITICAL'],
    
    # MASSIVE sequence parameters
    'sequence_length': 180,       # 3 minutes of history
    'prediction_horizon': 90,     # Predict state 90s ahead
    
    # 🔥 EXTREME architecture - optimized for 80GB GPU
    'model_dim': 256,             # Reduced from 512
    'num_heads': 8,               # Attention heads
    'num_transformer_layers': 4,  # Reduced from 6
    'lstm_hidden': 256,           # Reduced from 512
    'lstm_layers': 2,             # Reduced from 3
    'dropout': 0.3,
    
    # 🔥 Batch size - balanced for memory
    'batch_size': 2048,           # Reduced from 4096
    'accumulation_steps': 2,      # Effective batch: 4096
    
    # Training
    'epochs': 500,
    'learning_rate': 0.001,
    'patience': 50,
    'weight_decay': 1e-4,
    
    # Ensemble - single model to save memory
    'num_ensemble': 1,            # Reduced from 3
    
    # Data augmentation
    'augment': True,
    'noise_std': 0.05,            # Add noise to training data
    
    # Smoothing for state detection
    'smooth_window': 10,
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
print("⚡ EXTREME CLASSIFICATION CONFIGURATION ⚡")
print("="*80)
print(f"  📊 Sequence Length:     {CONFIG['sequence_length']}s (3 min history)")
print(f"  🎯 Prediction Horizon:  {CONFIG['prediction_horizon']}s ahead")
print(f"  🏗️  Transformer Dim:     {CONFIG['model_dim']}")
print(f"  🔢 Attention Heads:     {CONFIG['num_heads']}")
print(f"  📚 Transformer Layers:  {CONFIG['num_transformer_layers']}")
print(f"  🧠 LSTM Hidden:         {CONFIG['lstm_hidden']}")
print(f"  📦 Batch Size:          {CONFIG['batch_size']} (effective: {CONFIG['batch_size'] * CONFIG['accumulation_steps']})")
print(f"  🎭 Ensemble Models:     {CONFIG['num_ensemble']}")
print(f"  🏷️  Classes:             {CONFIG['class_names']}")
print("="*80)

# =============================================================================
# HYBRID TRANSFORMER-LSTM MODEL
# =============================================================================
class TransformerLSTMClassifier(nn.Module):
    """
    Hybrid architecture combining:
    - Transformer for global pattern recognition
    - LSTM for sequential dependencies
    - Both feed into classification head
    """
    
    def __init__(self, input_size, model_dim, num_heads, num_transformer_layers,
                 lstm_hidden, lstm_layers, num_classes, dropout=0.3, seq_len=180):
        super().__init__()
        
        self.model_dim = model_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_size, model_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, model_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=model_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(model_dim, model_dim // 4),
            nn.Tanh(),
            nn.Linear(model_dim // 4, 1)
        )
        
        # Combine transformer + LSTM outputs
        combined_size = model_dim + lstm_hidden * 2  # Transformer + BiLSTM
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_size),
            nn.Linear(combined_size, combined_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(combined_size // 2, combined_size // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(combined_size // 4, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (batch, seq, model_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer branch
        transformer_out = self.transformer(x)  # (batch, seq, model_dim)
        
        # Attention pooling for transformer
        attn_weights = torch.softmax(self.attention(transformer_out), dim=1)
        transformer_pooled = torch.sum(transformer_out * attn_weights, dim=1)  # (batch, model_dim)
        
        # LSTM branch
        lstm_out, _ = self.lstm(x)  # (batch, seq, lstm_hidden*2)
        lstm_pooled = lstm_out[:, -1, :]  # Take last timestep (batch, lstm_hidden*2)
        
        # Combine branches
        combined = torch.cat([transformer_pooled, lstm_pooled], dim=-1)
        
        # Classify
        logits = self.classifier(combined)  # (batch, num_classes)
        
        return logits

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
def bytes_to_state(value, thresholds):
    """Convert byte value to traffic state using dynamic thresholds"""
    if value < thresholds['NORMAL']:
        return 0  # NORMAL
    elif value < thresholds['ELEVATED']:
        return 1  # ELEVATED
    elif value < thresholds['HIGH']:
        return 2  # HIGH
    else:
        return 3  # CRITICAL

def calculate_dynamic_thresholds(data, percentiles):
    """Calculate thresholds from actual data distribution"""
    thresholds = {
        'NORMAL': np.percentile(data, percentiles['NORMAL']),
        'ELEVATED': np.percentile(data, percentiles['ELEVATED']),
        'HIGH': np.percentile(data, percentiles['HIGH']),
    }
    return thresholds

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
    
    # Smooth for state detection
    smoothed = pd.Series(raw_data).rolling(
        window=config['smooth_window'], 
        center=True,
        min_periods=1
    ).mean().values
    
    # DYNAMIC THRESHOLDS - calculated from THIS scenario's data!
    if config.get('use_dynamic_thresholds', True):
        thresholds = calculate_dynamic_thresholds(smoothed, config['percentiles'])
        print(f"  🎯 Dynamic thresholds (from data percentiles):")
        print(f"     NORMAL:   < {thresholds['NORMAL']:,.0f} bytes ({config['percentiles']['NORMAL']}th percentile)")
        print(f"     ELEVATED: < {thresholds['ELEVATED']:,.0f} bytes ({config['percentiles']['ELEVATED']}th percentile)")
        print(f"     HIGH:     < {thresholds['HIGH']:,.0f} bytes ({config['percentiles']['HIGH']}th percentile)")
        print(f"     CRITICAL: >= {thresholds['HIGH']:,.0f} bytes")
    else:
        thresholds = config['thresholds']
    
    # Convert to states
    states = np.array([bytes_to_state(v, thresholds) for v in smoothed])
    
    # Count class distribution
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
        # Target is the dominant state in the prediction window
        future_states = states[i + seq_len:i + seq_len + pred_horizon]
        # Use the state at the END of prediction horizon (90s ahead)
        y.append(states[i + seq_len + pred_horizon - 1])
    
    return np.array(X), np.array(y)

class TrafficDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.05):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and self.training:
            # Add noise during training
            noise = torch.randn_like(x) * self.noise_std * x.std()
            x = x + noise
        return x, self.y[idx]
    
    @property
    def training(self):
        return self.augment

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_classifier(scenario, data, states, config, model_idx=0):
    """Train a single classifier model"""
    
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
    
    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=4)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"  ⚖️ Class weights: {class_weights.cpu().numpy()}")
    
    # DataLoaders with large batch
    train_dataset = TrafficDataset(X_train, y_train, augment=config['augment'], noise_std=config['noise_std'])
    val_dataset = TrafficDataset(X_val, y_val)
    test_dataset = TrafficDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             num_workers=4, pin_memory=True)
    
    # Model
    model = TransformerLSTMClassifier(
        input_size=1,
        model_dim=config['model_dim'],
        num_heads=config['num_heads'],
        num_transformer_layers=config['num_transformer_layers'],
        lstm_hidden=config['lstm_hidden'],
        lstm_layers=config['lstm_layers'],
        num_classes=len(config['class_names']),
        dropout=config['dropout'],
        seq_len=seq_len
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model params: {total_params:,}")
    print(f"  📦 GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                           weight_decay=config['weight_decay'])
    
    # Cosine scheduler with warmup
    def warmup_cosine_schedule(epoch, warmup=10, total=config['epochs']):
        if epoch < warmup:
            return epoch / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (total - warmup)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_schedule)
    
    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_state = None
    
    print(f"\n  🏋️ Training model {model_idx+1}/{config['num_ensemble']}...")
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
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
        
        # Early stopping on accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            gpu_mem = torch.cuda.memory_allocated()/1e9
            print(f"     Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
                  f"LR: {lr:.2e} | GPU: {gpu_mem:.1f}GB")
        
        if patience_counter >= config['patience']:
            print(f"  ⏹️ Early stop at epoch {epoch+1}")
            break
    
    # Restore best model
    model.load_state_dict(best_state)
    
    # Evaluate on test set
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
    
    # Cleanup GPU memory
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
    """Plot classification results"""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'STATE CLASSIFIER: {scenario.upper()} (Ensemble of {len(results)})', 
                 fontsize=14, fontweight='bold')
    
    # 1. Training curves (first model)
    ax1 = plt.subplot(2, 2, 1)
    result = results[0]
    ax1.plot(result['train_accs'], 'b-', label='Train Acc', alpha=0.7)
    ax1.plot(result['val_accs'], 'r-', label='Val Acc', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Confusion matrix (ensemble average)
    ax2 = plt.subplot(2, 2, 2)
    # Combine predictions from all models
    all_preds = np.stack([r['predictions'] for r in results], axis=0)
    ensemble_preds = np.apply_along_axis(lambda x: np.bincount(x, minlength=4).argmax(), 0, all_preds)
    targets = results[0]['targets']
    
    cm = confusion_matrix(targets, ensemble_preds)
    im = ax2.imshow(cm, cmap='Blues')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(config['class_names'], rotation=45)
    ax2.set_yticklabels(config['class_names'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix (Ensemble)')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, cm[i, j], ha='center', va='center', fontsize=10,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    plt.colorbar(im, ax=ax2)
    
    # 3. Per-class accuracy
    ax3 = plt.subplot(2, 2, 3)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    colors = ['#27ae60' if a > 0.7 else '#f39c12' if a > 0.5 else '#e74c3c' for a in per_class_acc]
    bars = ax3.bar(config['class_names'], per_class_acc, color=colors, edgecolor='black')
    ax3.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Per-Class Accuracy')
    ax3.set_ylim(0, 1)
    for bar, val in zip(bars, per_class_acc):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=10)
    
    # 4. Summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    ensemble_acc = accuracy_score(targets, ensemble_preds)
    ensemble_f1 = f1_score(targets, ensemble_preds, average='weighted')
    
    status = '🏆 EXCELLENT' if ensemble_acc > 0.8 else '✨ GOOD' if ensemble_acc > 0.6 else '⚠️ OK'
    color = '#27ae60' if ensemble_acc > 0.8 else '#f39c12' if ensemble_acc > 0.6 else '#e74c3c'
    
    text = f"""
    ════════════════════════════════════
       ENSEMBLE CLASSIFICATION RESULTS
    ════════════════════════════════════
    
    📊 Accuracy:      {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)
    📈 F1 Score:      {ensemble_f1:.4f}
    
    🎭 Models:        {len(results)} (ensemble voting)
    ⏱️  Epochs:        ~{np.mean([r['epochs_trained'] for r in results]):.0f}
    
    🔮 Predicts:      {config['prediction_horizon']}s ahead
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

# =============================================================================
# MAIN TRAINING
# =============================================================================
all_results = {}
total_start = time.time()

for idx, (scenario, pattern) in enumerate(SCENARIOS.items(), 1):
    print(f"\n\n{'='*80}")
    print(f"[{idx}/{len(SCENARIOS)}] TRAINING CLASSIFIER: {scenario.upper()}")
    print('='*80)
    
    try:
        # CLEANUP GPU MEMORY before each scenario
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            print(f"  🧹 GPU memory cleared. Available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")
        
        # Load data
        data, states, thresholds = load_and_prepare_classification_data(scenario, pattern, DATA_FOLDER, CONFIG)
        
        # Train ensemble
        scenario_results = []
        scenario_start = time.time()
        
        for m_idx in range(CONFIG['num_ensemble']):
            print(f"\n  --- Ensemble Model {m_idx+1}/{CONFIG['num_ensemble']} ---")
            result = train_classifier(scenario, data, states, CONFIG, model_idx=m_idx)
            scenario_results.append(result)
            print(f"  ✓ Accuracy: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}")
        
        training_time = time.time() - scenario_start
        
        # Ensemble metrics
        all_preds = np.stack([r['predictions'] for r in scenario_results], axis=0)
        ensemble_preds = np.apply_along_axis(lambda x: np.bincount(x, minlength=4).argmax(), 0, all_preds)
        targets = scenario_results[0]['targets']
        
        ensemble_acc = accuracy_score(targets, ensemble_preds)
        ensemble_f1 = f1_score(targets, ensemble_preds, average='weighted')
        
        print(f"\n  📊 ENSEMBLE RESULTS:")
        print(f"     Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")
        print(f"     F1 Score: {ensemble_f1:.4f}")
        print(f"     Time:     {training_time/60:.1f} min")
        
        # Save first model (for inference)
        model_path = f"{MODELS_FOLDER}/{scenario}_classifier.pt"
        scaler_path = f"{MODELS_FOLDER}/{scenario}_classifier_scaler.pkl"
        
        torch.save({
            'model': scenario_results[0]['model'],
            'model_state_dict': scenario_results[0]['model'].state_dict(),
            'config': {
                'sequence_length': CONFIG['sequence_length'],
                'prediction_horizon': CONFIG['prediction_horizon'],
                'class_names': CONFIG['class_names'],
                'thresholds': thresholds,  # DYNAMIC thresholds for this scenario!
                'percentiles': CONFIG['percentiles'],
                'model_type': 'TransformerLSTMClassifier'
            }
        }, model_path)
        joblib.dump(scenario_results[0]['scaler'], scaler_path)
        print(f"  ✓ Saved: {model_path}")
        
        # Save visualization
        vis_path = f"{VISUALS_FOLDER}/{scenario}_classifier_training.png"
        plot_classifier_results(scenario, scenario_results, CONFIG, vis_path)
        print(f"  ✓ Plot: {vis_path}")
        
        all_results[scenario] = {
            'accuracy': ensemble_acc,
            'f1_score': ensemble_f1,
            'training_time_min': training_time / 60,
            'epochs': np.mean([r['epochs_trained'] for r in scenario_results])
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# Final summary
print("\n\n" + "="*80)
print("🏁 CLASSIFICATION TRAINING COMPLETE")
print("="*80)

if all_results:
    print(f"\n{'Scenario':<12} {'Accuracy':<10} {'F1':<10} {'Time':<8} {'Status'}")
    print("-"*60)
    for s, m in all_results.items():
        status = "🏆" if m['accuracy'] > 0.8 else "✨" if m['accuracy'] > 0.6 else "⚠️"
        print(f"{s:<12} {m['accuracy']:<10.4f} {m['f1_score']:<10.4f} {m['training_time_min']:<8.1f} {status}")
    
    avg_acc = np.mean([m['accuracy'] for m in all_results.values()])
    print("-"*60)
    print(f"{'Average:':<12} {avg_acc:.4f}")
    
    print(f"\n🎯 Classes: {CONFIG['class_names']}")
    print(f"🔮 Predicts state {CONFIG['prediction_horizon']}s ahead")

print(f"\n⏱️ Total time: {(time.time()-total_start)/60:.1f} minutes")
print(f"\n📁 Models: {MODELS_FOLDER}/*_classifier.pt")
print(f"📊 Plots: {VISUALS_FOLDER}/*_classifier_training.png")

gpu_mem = torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0
print(f"💾 Peak GPU memory: {gpu_mem:.1f} GB")
print("="*80)

# src/ml_models/seq2seq_predictor.py
"""
Seq2Seq Traffic Predictor for Graph Visualization

Loads trained models and predicts 60 future traffic values.
Supports ALL training architectures:
- FastPredictor (train_fast_regression.py) 
- A100Seq2Seq (train_on_a100.py)
- FastSeq2Seq (train_seq2seq_predictor.py)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import os
import joblib


# Architecture from train_fast_regression.py
class FastPredictor(nn.Module):
    """CNN + BiLSTM with MaxPool - direct 60-output (train_fast_regression.py)"""
    
    def __init__(self, config):
        super().__init__()
        hidden = config.get('hidden', config.get('hidden_size', 256))
        pred_len = config.get('prediction_horizon', 60)
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        self.lstm = nn.LSTM(128, hidden, 2, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, pred_len)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h)


# Architecture from A100 training
class A100Seq2Seq(nn.Module):
    """CNN + BiLSTM (A100 training - no MaxPool)"""
    
    def __init__(self, config):
        super().__init__()
        hidden = config.get('hidden_size', 256)
        layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.3)
        pred_horizon = config.get('prediction_horizon', 60)
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(128, hidden, layers, batch_first=True, 
                           dropout=dropout if layers > 1 else 0, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_horizon)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


class Seq2SeqPredictor:
    """
    Traffic Predictor for Graph Visualization.
    Auto-detects model architecture from checkpoint.
    """
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Seq2SeqPredictor using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        state_dict = checkpoint['model_state_dict']
        
        self.sequence_length = config.get('sequence_length', 90)
        self.prediction_horizon = config.get('prediction_horizon', 60)
        
        # Detect architecture from checkpoint
        model_type = checkpoint.get('model_type', None)
        has_maxpool = any('conv.2' in k for k in state_dict.keys())  # MaxPool is index 2
        has_conv = any('conv' in k for k in state_dict.keys())
        
        if model_type == 'FastPredictor' or has_maxpool:
            print("  Detected FastPredictor architecture (CNN+MaxPool+LSTM)")
            self.model = FastPredictor(config).to(self.device)
        elif has_conv:
            print("  Detected A100 architecture (CNN+LSTM)")
            self.model = A100Seq2Seq(config).to(self.device)
        else:
            raise ValueError("Unknown model architecture")
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"✓ Seq2Seq loaded: {model_path}")
        print(f"  Input: {self.sequence_length}s → Output: {self.prediction_horizon} values")
        
        # Load scaler
        self.scaler = None
        if scaler_path is None:
            scaler_path = model_path.replace('.pt', '_scaler.pkl')
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded: {scaler_path}")
    
    def predict(self, historical_data: np.ndarray) -> np.ndarray:
        """
        Predict 60 future traffic values from historical data.
        """
        # Ensure correct shape
        if historical_data.ndim == 1:
            historical_data = historical_data.reshape(-1, 1)
        
        # Pad or truncate
        if len(historical_data) < self.sequence_length:
            padding = np.full((self.sequence_length - len(historical_data), 1), 
                            historical_data[0, 0])
            historical_data = np.vstack([padding, historical_data])
        elif len(historical_data) > self.sequence_length:
            historical_data = historical_data[-self.sequence_length:]
        
        # Calculate input stats BEFORE normalization (in original bytes/s)
        input_mean = float(np.mean(historical_data))
        input_max = float(np.max(historical_data))
        
        # Normalize
        if self.scaler is not None:
            data_scaled = self.scaler.transform(historical_data.astype(np.float32))
        else:
            data_scaled = historical_data.astype(np.float32)
        
        # Predict
        x = torch.FloatTensor(data_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
        
        # Get predictions
        predictions = output[0].cpu().numpy() if output.dim() == 2 else output.cpu().numpy()
        predictions = predictions.reshape(-1, 1)
        
        # Inverse transform
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        
        predictions = predictions.flatten()
        
        # CRITICAL: If input traffic is very low, scale predictions DOWN
        # This prevents the model from "hallucinating" patterns when there's no traffic
        LOW_TRAFFIC_THRESHOLD = 5000  # 5 KB/s - be aggressive
        
        if input_mean < LOW_TRAFFIC_THRESHOLD:
            # Very low traffic - predictions should stay near actual input
            scale = max(0.001, input_mean / LOW_TRAFFIC_THRESHOLD)
            # Blend heavily toward actual input mean
            predictions = predictions * scale * 0.1 + input_mean * 0.9
        
        return np.clip(predictions, 0, None)
    
    def get_info(self) -> Dict:
        return {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'device': str(self.device)
        }


if __name__ == '__main__':
    import glob
    models = glob.glob('models/*_seq2seq_3050.pt') + glob.glob('models/seq2seqA100/*_seq2seq_3050.pt')
    if models:
        print(f"Testing: {models[0]}")
        p = Seq2SeqPredictor(models[0])
        fake = np.random.exponential(500000, 90)
        pred = p.predict(fake)
        print(f"Output: {pred.shape}, range: {pred.min():.0f} - {pred.max():.0f}")

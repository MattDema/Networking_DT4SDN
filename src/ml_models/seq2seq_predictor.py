# src/ml_models/seq2seq_predictor.py
"""
Seq2Seq Traffic Predictor - Sequence-to-Sequence Time Series Forecasting

This module handles:
1. Loading multi-scenario Seq2Seq models (Normal, Burst, DDoS, etc.)
2. Auto-detecting architecture (CNN+LSTM vs FastLSTM)
3. Generating 60-second forward predictions for graph visualization
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import os
import joblib


# Architecture used by A100 training (CNN + BiLSTM)
class A100Seq2Seq(nn.Module):
    """CNN + BiLSTM Seq2Seq (matches A100 training architecture)"""
    
    def __init__(self, config):
        super().__init__()
        hidden = config.get('hidden_size', 256)
        layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.3)
        pred_horizon = config.get('prediction_horizon', 60)
        
        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_horizon)
        )
        
        self.prediction_horizon = pred_horizon
    
    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = x.permute(0, 2, 1)  # (batch, 1, seq_len)
        x = self.conv(x)  # (batch, 128, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 128)
        
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Use last output
        last_out = lstm_out[:, -1, :]  # (batch, hidden*2)
        output = self.fc(last_out)  # (batch, pred_horizon)
        
        return output


# Architecture used by laptop training (FastSeq2Seq)
class FastSeq2Seq(nn.Module):
    """Fast LSTM Encoder-Decoder (laptop training architecture)"""
    
    def __init__(self, config):
        super().__init__()
        hidden = config.get('hidden_size', 128)
        layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.2)
        
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
            bidirectional=True
        )
        
        self.bridge_h = nn.Linear(hidden * 2, hidden)
        self.bridge_c = nn.Linear(hidden * 2, hidden)
        
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden, 1)
        
        self.prediction_horizon = config.get('prediction_horizon', 60)
        self.num_layers = layers
        self.hidden_size = hidden
    
    def forward(self, src, target=None, teacher_forcing=0.0):
        batch_size = src.size(0)
        
        _, (h_enc, c_enc) = self.encoder(src)
        
        h_enc = h_enc.view(self.num_layers, 2, batch_size, self.hidden_size)
        c_enc = c_enc.view(self.num_layers, 2, batch_size, self.hidden_size)
        
        h_cat = torch.cat([h_enc[:, 0], h_enc[:, 1]], dim=2)
        c_cat = torch.cat([c_enc[:, 0], c_enc[:, 1]], dim=2)
        
        h_dec = self.bridge_h(h_cat)
        c_dec = self.bridge_c(c_cat)
        
        last_val = src[:, -1:, :]
        decoder_input = last_val.repeat(1, self.prediction_horizon, 1)
        
        decoder_output, _ = self.decoder(decoder_input, (h_dec, c_dec))
        output = self.fc(decoder_output).squeeze(-1)
        
        return output


class Seq2SeqPredictor:
    """
    Seq2Seq Traffic Predictor for Graph Visualization.
    Auto-detects model architecture (A100 or laptop).
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
        
        # Auto-detect architecture from state_dict keys
        has_conv = any('conv' in k for k in state_dict.keys())
        has_encoder = any('encoder' in k for k in state_dict.keys())
        
        if has_conv:
            print("  Detected A100 architecture (CNN+LSTM)")
            self.model = A100Seq2Seq(config).to(self.device)
        elif has_encoder:
            print("  Detected FastSeq2Seq architecture")
            self.model = FastSeq2Seq(config).to(self.device)
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
        
        Args:
            historical_data: Recent traffic (bytes/s), shape (sequence_length,) or (N,)
        
        Returns:
            np.ndarray: 60 predicted future values (bytes/s)
        """
        # Ensure correct shape
        if historical_data.ndim == 1:
            historical_data = historical_data.reshape(-1, 1)
        
        # Pad or truncate to sequence_length
        if len(historical_data) < self.sequence_length:
            padding = np.full((self.sequence_length - len(historical_data), 1), 
                            historical_data[0, 0])
            historical_data = np.vstack([padding, historical_data])
        elif len(historical_data) > self.sequence_length:
            historical_data = historical_data[-self.sequence_length:]
        
        # Normalize
        if self.scaler is not None:
            data_scaled = self.scaler.transform(historical_data.astype(np.float32))
        else:
            data_scaled = historical_data.astype(np.float32)
        
        # Convert to tensor
        x = torch.FloatTensor(data_scaled).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(x)
        
        # Get predictions
        if output.dim() == 2:
            predictions = output[0].cpu().numpy()
        else:
            predictions = output.cpu().numpy()
        
        predictions = predictions.reshape(-1, 1)
        
        # Inverse transform
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        
        # Clip to non-negative
        predictions = np.clip(predictions.flatten(), 0, None)
        
        return predictions
    
    def get_info(self) -> Dict:
        return {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'device': str(self.device)
        }


class Seq2SeqManager:
    """
    Manager for multiple seq2seq models.
    Allows switching between different scenario models.
    """
    
    SCENARIOS = ['normal', 'burst', 'congestion', 'ddos', 'mixed']
    
    def __init__(self, models_dir: str):
        """
        Initialize manager and discover available models.
        
        Args:
            models_dir: Directory containing seq2seq model files
        """
        self.models_dir = models_dir
        self.available_models = {}
        self.current_model = None
        self.current_scenario = None
        
        # Discover available models
        self._discover_models()
        
        # Load default model (mixed or first available)
        if 'mixed' in self.available_models:
            self.switch_model('mixed')
        elif self.available_models:
            self.switch_model(list(self.available_models.keys())[0])
    
    def _discover_models(self):
        """Find all available seq2seq models in the directory."""
        import glob
        
        # Check both regular models dir and seq2seqA100 subdir
        patterns = [
            os.path.join(self.models_dir, 'seq2seqA100', '*_seq2seq_3050.pt'),
            os.path.join(self.models_dir, '*_seq2seq_3050.pt')
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            for f in files:
                basename = os.path.basename(f)
                for scenario in self.SCENARIOS:
                    if basename.startswith(scenario):
                        self.available_models[scenario] = f
                        break
        
        print(f"[Seq2SeqManager] Found models: {list(self.available_models.keys())}")
    
    def switch_model(self, scenario: str) -> bool:
        """
        Switch to a different scenario model.
        
        Args:
            scenario: One of 'normal', 'burst', 'congestion', 'ddos', 'mixed'
        
        Returns:
            True if switch successful, False otherwise
        """
        if scenario not in self.available_models:
            print(f"[Seq2SeqManager] Model '{scenario}' not available")
            return False
        
        if scenario == self.current_scenario:
            return True  # Already loaded
        
        try:
            model_path = self.available_models[scenario]
            self.current_model = Seq2SeqPredictor(model_path)
            self.current_scenario = scenario
            print(f"[Seq2SeqManager] Switched to: {scenario}")
            return True
        except Exception as e:
            print(f"[Seq2SeqManager] Error loading {scenario}: {e}")
            return False
    
    def predict(self, historical_data: np.ndarray) -> Optional[np.ndarray]:
        """Predict using current model."""
        if self.current_model is None:
            return None
        return self.current_model.predict(historical_data)
    
    def get_available_models(self) -> list:
        """Return list of available models."""
        return list(self.available_models.keys())
    
    def get_current_model(self) -> Optional[str]:
        """Return current model name."""
        return self.current_scenario
    
    def get_info(self) -> Dict:
        """Return manager info."""
        return {
            'available_models': list(self.available_models.keys()),
            'current_model': self.current_scenario,
            'model_info': self.current_model.get_info() if self.current_model else None
        }


def test_predictor():
    """Test the seq2seq predictor."""
    import glob
    
    model_dir = 'models/seq2seqA100'
    models = glob.glob(f'{model_dir}/*_seq2seq_3050.pt')
    
    if not models:
        model_dir = 'models'
        models = glob.glob(f'{model_dir}/*_seq2seq_3050.pt')
    
    if not models:
        print("No seq2seq models found!")
        return
    
    model_path = models[0]
    print(f"Testing with: {model_path}")
    
    predictor = Seq2SeqPredictor(model_path)
    
    # Generate fake traffic data
    np.random.seed(42)
    fake_traffic = np.random.exponential(500000, size=90)
    
    print("\nTest prediction:")
    predictions = predictor.predict(fake_traffic)
    print(f"  Input shape: {fake_traffic.shape}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Output range: {predictions.min():.0f} - {predictions.max():.0f}")
    
    print("\n✓ Seq2SeqPredictor test passed!")


if __name__ == '__main__':
    test_predictor()


# src/ml_models/seq2seq_predictor.py
"""
Seq2Seq Traffic Predictor for Graph Visualization

Loads trained seq2seq models and predicts 60 future traffic values.
Used alongside the state classifier for realistic graph predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import os
import joblib


class FastSeq2Seq(nn.Module):
    """Fast LSTM Encoder-Decoder (matches training architecture)"""
    
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
    
    Usage:
        predictor = Seq2SeqPredictor('models/seq2seqA100/mixed_seq2seq_3050.pt')
        future_values = predictor.predict(traffic_history)
        # Returns: array of 60 predicted traffic values (bytes/s)
    """
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Seq2SeqPredictor using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        
        self.sequence_length = config.get('sequence_length', 90)
        self.prediction_horizon = config.get('prediction_horizon', 60)
        
        # Build model
        self.model = FastSeq2Seq(config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
        
        # Inverse transform
        predictions = output[0].cpu().numpy().reshape(-1, 1)
        
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

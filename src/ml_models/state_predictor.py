# src/ml_models/state_predictor.py
"""
State Predictor - Classification-based traffic state prediction

Loads trained classifier models and predicts traffic state:
- NORMAL, ELEVATED, HIGH, CRITICAL

Also estimates bandwidth from predicted state using saved thresholds.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import joblib


class BiLSTMClassifier(nn.Module):
    """BiLSTM Classifier - must match training architecture"""
    
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
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1)
        )
        
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
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.classifier(context)


class StatePredictor:
    """
    Traffic State Predictor using trained classification models.
    
    Usage:
        predictor = StatePredictor('models/mixed_classifier_3050.pt')
        result = predictor.predict(traffic_history)
        # result = {'state': 'HIGH', 'state_id': 2, 'confidence': 0.85, 
        #           'estimated_bandwidth': 2500000, 'probabilities': [...]}
    """
    
    CLASS_NAMES = ['NORMAL', 'ELEVATED', 'HIGH', 'CRITICAL']
    CLASS_COLORS = {
        'NORMAL': '#27ae60',    # Green
        'ELEVATED': '#f39c12',  # Orange
        'HIGH': '#e67e22',      # Dark Orange
        'CRITICAL': '#e74c3c'   # Red
    }
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Initialize predictor with trained classifier model.
        
        Args:
            model_path: Path to saved model (.pt file)
            scaler_path: Path to saved scaler (.pkl file)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"StatePredictor using device: {self.device}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        
        # Extract config
        self.sequence_length = config['sequence_length']
        self.prediction_horizon = config['prediction_horizon']
        self.class_names = config.get('class_names', self.CLASS_NAMES)
        self.thresholds = config.get('thresholds', {})
        
        # Rebuild model architecture
        lstm_hidden = config.get('lstm_hidden', 128)
        lstm_layers = config.get('lstm_layers', 2)
        
        self.model = BiLSTMClassifier(
            input_size=1,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            num_classes=len(self.class_names),
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded: {model_path}")
        print(f"  Sequence length: {self.sequence_length}s")
        print(f"  Prediction horizon: {self.prediction_horizon}s")
        print(f"  Thresholds: {self.thresholds}")
        
        # Load scaler
        self.scaler = None
        if scaler_path is None:
            scaler_path = model_path.replace('.pt', '_scaler.pkl')
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded: {scaler_path}")
    
    def predict(self, historical_data: np.ndarray) -> Dict:
        """
        Predict traffic state from historical data.
        
        Args:
            historical_data: Recent traffic data (bytes per second)
                           Shape: (sequence_length,) or (sequence_length, 1)
        
        Returns:
            dict with keys:
                - state: str ('NORMAL', 'ELEVATED', 'HIGH', 'CRITICAL')
                - state_id: int (0-3)
                - confidence: float (0-1)
                - probabilities: list of 4 floats
                - estimated_bandwidth: float (bytes/s)
                - color: str (hex color for visualization)
        """
        # Reshape if needed
        if historical_data.ndim == 1:
            historical_data = historical_data.reshape(-1, 1)
        
        # Validate length
        if len(historical_data) < self.sequence_length:
            # Pad with first value if too short
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
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            state_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0, state_id].item()
        
        probabilities = probs[0].cpu().numpy().tolist()
        state = self.class_names[state_id]
        
        # Estimate bandwidth from state AND recent traffic pattern
        avg_recent = float(np.mean(historical_data))
        estimated_bw = self._estimate_bandwidth(state_id, avg_recent)
        
        return {
            'state': state,
            'state_id': state_id,
            'confidence': round(confidence, 4),
            'probabilities': [round(p, 4) for p in probabilities],
            'estimated_bandwidth': estimated_bw,
            'color': self.CLASS_COLORS.get(state, '#666666'),
            'prediction_horizon': self.prediction_horizon
        }
    
    def _estimate_bandwidth(self, state_id: int, current_avg: float = 0) -> float:
        """
        Estimate future bandwidth based on predicted state AND current traffic.
        
        If traffic is low and state is NORMAL, prediction stays low.
        If state changes, we project toward that state's expected range.
        """
        
        if not self.thresholds:
            # No thresholds saved, use current average
            return current_avg
        
        th_normal = self.thresholds.get('NORMAL', 500000)
        th_elevated = self.thresholds.get('ELEVATED', 1500000)
        th_high = self.thresholds.get('HIGH', 2500000)
        
        if state_id == 0:  # NORMAL
            # For NORMAL state, use actual traffic (don't inflate to threshold)
            # But ensure it's within NORMAL range
            return min(current_avg, th_normal * 0.8)
        elif state_id == 1:  # ELEVATED
            # Blend between current and ELEVATED range
            target = (th_normal + th_elevated) / 2
            return max(current_avg, target * 0.7)  # At least 70% of range
        elif state_id == 2:  # HIGH
            target = (th_elevated + th_high) / 2
            return max(current_avg, target * 0.7)
        else:  # CRITICAL
            # CRITICAL means expect high traffic
            return max(current_avg, th_high)
    
    def get_state_info(self, state: str) -> Dict:
        """Get information about a state including suggested actions."""
        actions = {
            'NORMAL': {
                'level': 0,
                'description': 'Traffic within normal range',
                'action': 'No action required',
                'color': self.CLASS_COLORS['NORMAL']
            },
            'ELEVATED': {
                'level': 1,
                'description': 'Traffic above normal, monitor closely',
                'action': 'Monitor for potential issues',
                'color': self.CLASS_COLORS['ELEVATED']
            },
            'HIGH': {
                'level': 2,
                'description': 'High traffic detected',
                'action': 'Consider load balancing or traffic shaping',
                'color': self.CLASS_COLORS['HIGH']
            },
            'CRITICAL': {
                'level': 3,
                'description': 'Critical traffic level - possible congestion/attack',
                'action': 'Immediate mitigation required: reroute or rate-limit',
                'color': self.CLASS_COLORS['CRITICAL']
            }
        }
        return actions.get(state, actions['NORMAL'])
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'class_names': self.class_names,
            'thresholds': self.thresholds,
            'device': str(self.device)
        }


class StateManager:
    """
    Manager for multiple classifier models.
    Allows switching between different scenario models.
    """
    
    SCENARIOS = ['normal', 'burst', 'congestion', 'ddos', 'mixed']
    
    def __init__(self, models_dir: str):
        """
        Initialize manager and discover available models.
        
        Args:
            models_dir: Directory containing classifier model files
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
        """Find all available classifier models in the directory."""
        import glob
        
        pattern = os.path.join(self.models_dir, '*_classifier_3050.pt')
        files = glob.glob(pattern)
        
        for f in files:
            basename = os.path.basename(f)
            for scenario in self.SCENARIOS:
                if basename.startswith(scenario):
                    self.available_models[scenario] = f
                    break
        
        print(f"[StateManager] Found models: {list(self.available_models.keys())}")
    
    def switch_model(self, scenario: str) -> bool:
        """
        Switch to a different scenario model.
        
        Args:
            scenario: One of 'normal', 'burst', 'congestion', 'ddos', 'mixed'
        
        Returns:
            True if switch successful, False otherwise
        """
        if scenario not in self.available_models:
            print(f"[StateManager] Model '{scenario}' not available")
            return False
        
        if scenario == self.current_scenario:
            return True  # Already loaded
        
        try:
            model_path = self.available_models[scenario]
            self.current_model = StatePredictor(model_path)
            self.current_scenario = scenario
            print(f"[StateManager] Switched to: {scenario}")
            return True
        except Exception as e:
            print(f"[StateManager] Error loading {scenario}: {e}")
            return False
    
    def predict(self, historical_data: np.ndarray, current_avg: float = None) -> Optional[Dict]:
        """Predict using current model."""
        if self.current_model is None:
            return None
        return self.current_model.predict(historical_data)
    
    @property
    def sequence_length(self) -> int:
        """Get sequence length from current model."""
        if self.current_model:
            return self.current_model.sequence_length
        return 90
    
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
            'model_info': self.current_model.get_model_info() if self.current_model else None
        }


def test_predictor():
    """Test the state predictor."""
    import glob
    
    # Find a model
    models = glob.glob('models/*_classifier_3050.pt')
    if not models:
        print("No classifier models found!")
        return
    
    model_path = models[0]
    print(f"Testing with: {model_path}")
    
    predictor = StatePredictor(model_path)
    
    # Generate fake traffic data (90 seconds)
    np.random.seed(42)
    fake_traffic = np.random.exponential(1000000, size=90)
    
    print("\nTest predictions:")
    result = predictor.predict(fake_traffic)
    print(f"  State: {result['state']} (confidence: {result['confidence']:.2f})")
    print(f"  Estimated bandwidth: {result['estimated_bandwidth']:,.0f} bytes/s")
    print(f"  Probabilities: {result['probabilities']}")
    print(f"  Color: {result['color']}")
    
    print("\n✓ StatePredictor test passed!")


if __name__ == '__main__':
    test_predictor()


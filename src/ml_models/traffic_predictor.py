# src/ml_models/traffic_predictor.py
"""
Traffic Predictor - Main interface for traffic prediction (PyTorch version)
Loads trained model and makes predictions on live data
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
import os


class TrafficPredictor:
    """
    Traffic prediction interface using trained LSTM/GRU model (PyTorch)
    
    Usage:
        predictor = TrafficPredictor('models/traffic_model.pt')
        predictions = predictor.predict(recent_traffic_data)
    """
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved PyTorch model (.pt or .pth)
            scaler_path: Path to saved scaler (for data normalization)
        """
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = checkpoint['model']
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"✓ Model loaded from {model_path}")
        
        # Load scaler if provided
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            import joblib
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded from {scaler_path}")
        
        # Model configuration
        self.sequence_length = checkpoint['config']['sequence_length']
        self.num_features = checkpoint['config']['num_features']
        self.prediction_horizon = checkpoint['config']['prediction_horizon']
        
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Prediction horizon: {self.prediction_horizon}")
    
    def predict(self, historical_data: np.ndarray, return_confidence: bool = False) -> np.ndarray:
        """
        Predict future traffic based on historical data
        
        Args:
            historical_data: Recent traffic data
                            Shape: (sequence_length,) or (sequence_length, num_features)
            return_confidence: If True, return prediction intervals
        
        Returns:
            predictions: Predicted traffic for next N timesteps
        """
        # Reshape input if needed
        if historical_data.ndim == 1:
            historical_data = historical_data.reshape(-1, 1)
        
        # Validate input shape
        if len(historical_data) != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({len(historical_data)}) "
                f"doesn't match model expected length ({self.sequence_length})"
            )
        
        # Normalize if scaler is available
        if self.scaler is not None:
            historical_data = self.scaler.transform(historical_data)
        
        # Convert to PyTorch tensor
        # Shape: (1, sequence_length, num_features) - batch size of 1
        x = torch.FloatTensor(historical_data).unsqueeze(0).to(self.device)
        
        # Make prediction (no gradients needed for inference)
        with torch.no_grad():
            predictions = self.model(x)
        
        # Convert back to numpy
        predictions = predictions.cpu().numpy().squeeze(0)
        
        # Denormalize predictions if scaler was used
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        if return_confidence:
            # TODO: Implement prediction intervals
            return predictions, None
        
        return predictions
    
    def predict_next_frame(self, link_id: str, db_manager) -> dict:
        """
        Predict traffic for next frame given a link ID
        Fetches recent data from database automatically
        """
        import time
        
        # Fetch last N seconds of data for this link
        historical_data = db_manager.get_recent_traffic(
            link_id=link_id,
            duration_seconds=self.sequence_length
        )
        
        if len(historical_data) < self.sequence_length:
            raise ValueError(
                f"Insufficient historical data for {link_id}. "
                f"Need {self.sequence_length} samples, got {len(historical_data)}"
            )
        
        # Extract traffic values
        traffic_values = historical_data['bytes_sent'].values
        
        # Make prediction
        predictions = self.predict(traffic_values)
        
        return {
            'link_id': link_id,
            'predictions': predictions,
            'timestamp': time.time(),
            'horizon_seconds': self.prediction_horizon
        }
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            'sequence_length': self.sequence_length,
            'num_features': self.num_features,
            'prediction_horizon': self.prediction_horizon,
            'total_parameters': total_params,
            'model_type': self.model.__class__.__name__,
            'device': str(self.device)
        }


def test_predictor():
    """Test function to verify predictor works"""
    import numpy as np
    
    print("Testing TrafficPredictor...")
    
    # Create dummy model for testing
    predictor = TrafficPredictor('models/traffic_model.pt')
    
    # Generate dummy input (60 seconds of traffic)
    dummy_input = np.random.randint(1000, 3000, size=60)
    
    print(f"\nInput traffic (last 60s): {dummy_input[:10]}... (showing first 10)")
    
    # Make prediction
    predictions = predictor.predict(dummy_input)
    
    print(f"Predicted traffic (next {len(predictions)}s): {predictions[:10]}... (showing first 10)")
    print(f"\nPrediction range: [{predictions.min():.0f}, {predictions.max():.0f}] bytes")
    
    print("\n✓ Predictor test passed!")


if __name__ == '__main__':
    if os.path.exists('models/traffic_model.pt'):
        test_predictor()
    else:
        print("No trained model found. Please train a model first using model_trainer.py")

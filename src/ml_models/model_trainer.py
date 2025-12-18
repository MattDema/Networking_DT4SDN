# src/ml_models/model_trainer.py
"""
Model Trainer - Train LSTM/GRU models on historical traffic data (PyTorch)
Run this script to train new models on collected traffic data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
from typing import Tuple, List


class TrafficDataset(Dataset):
    """PyTorch Dataset for traffic time series"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


""" class TrafficLSTM(nn.Module):
    # LSTM model for traffic prediction
    
    def __init__(self, input_size: int, hidden_units: List[int], 
                 prediction_horizon: int, dropout: float = 0.2):
        super(TrafficLSTM, self).__init__()
        
        self.hidden_units = hidden_units
        self.num_layers = len(hidden_units)
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First layer
        self.lstm_layers.append(
            nn.LSTM(input_size, hidden_units[0], batch_first=True)
        )
        
        # Additional layers
        for i in range(1, len(hidden_units)):
            self.lstm_layers.append(
                nn.LSTM(hidden_units[i-1], hidden_units[i], batch_first=True)
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_units[-1], prediction_horizon)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)
        
        # Take last timestep output
        x = x[:, -1, :]  # (batch, hidden_units[-1])
        
        # Fully connected layer
        x = self.fc(x)  # (batch, prediction_horizon)
        
        # Reshape to (batch, prediction_horizon, 1)
        x = x.unsqueeze(-1)
        
        return x """

class TrafficLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_units: List[int], 
                 prediction_horizon: int, dropout: float = 0.2,
                 bidirectional: bool = True):  # NEW parameter
        super(TrafficLSTM, self).__init__()
        
        self.hidden_units = hidden_units
        self.num_layers = len(hidden_units)
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First layer
        self.lstm_layers.append(
            nn.LSTM(input_size, hidden_units[0], batch_first=True, 
                   bidirectional=bidirectional)  # Enable bidirectional
        )
        
        # Additional layers
        for i in range(1, len(hidden_units)):
            input_dim = hidden_units[i-1] * (2 if bidirectional else 1)  # Account for bidirectional
            self.lstm_layers.append(
                nn.LSTM(input_dim, hidden_units[i], batch_first=True,
                       bidirectional=bidirectional)
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (account for bidirectional)
        final_hidden_size = hidden_units[-1] * (2 if bidirectional else 1)
        self.fc = nn.Linear(final_hidden_size, prediction_horizon)
    
    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)
        
        x = x[:, -1, :]  # Take last timestep
        x = self.fc(x)
        x = x.unsqueeze(-1)
        
        return x


class TrafficGRU(nn.Module):
    """GRU model for traffic prediction"""
    
    def __init__(self, input_size: int, hidden_units: List[int], 
                 prediction_horizon: int, dropout: float = 0.2):
        super(TrafficGRU, self).__init__()
        
        self.hidden_units = hidden_units
        self.num_layers = len(hidden_units)
        
        # GRU layers
        self.gru_layers = nn.ModuleList()
        
        # First layer
        self.gru_layers.append(
            nn.GRU(input_size, hidden_units[0], batch_first=True)
        )
        
        # Additional layers
        for i in range(1, len(hidden_units)):
            self.gru_layers.append(
                nn.GRU(hidden_units[i-1], hidden_units[i], batch_first=True)
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_units[-1], prediction_horizon)
    
    def forward(self, x):
        for gru in self.gru_layers:
            x, _ = gru(x)
            x = self.dropout(x)
        
        x = x[:, -1, :]
        x = self.fc(x)
        x = x.unsqueeze(-1)
        
        return x


class TrafficModelTrainer:
    """Trains LSTM/GRU models for traffic prediction"""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 30,
                 model_type: str = 'lstm'):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type.lower()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_losses = []
        self.val_losses = []
        
        print(f"\nTrainer initialized:")
        print(f"  Sequence length: {sequence_length}s")
        print(f"  Prediction horizon: {prediction_horizon}s")
        print(f"  Model type: {model_type.upper()}")
    
    def load_data(self, data_path: str, target_column: str = 'bytes_sent'):
        """Load and preprocess training data"""
        print(f"\nLoading data from {data_path}...")
        
        df = pd.read_csv(data_path)
        print(f"  Loaded {len(df)} records")
        
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in data")
        
        data = df[target_column].values.reshape(-1, 1)
        print(f"  Using column: {target_column}")
        
        # Normalize
        data_normalized = self.scaler.fit_transform(data)
        print(f"  Normalized range: [{data_normalized.min():.3f}, {data_normalized.max():.3f}]")
        print(f"  Data range: [{data.min():.0f}, {data.max():.0f}]")
        
        # Create sequences
        X, y = self._create_sequences(data_normalized)
        print(f"  Created {len(X)} sequences")
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"  Train samples: {len(self.X_train)}")
        print(f"  Test samples: {len(self.X_test)}")
        print("✓ Data loaded successfully")
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def build_model(self, units: List[int] = [64, 32], dropout: float = 0.2):
        """Build LSTM or GRU model"""
        print(f"\nBuilding {self.model_type.upper()} model...")
        
        if self.model_type == 'lstm':
            self.model = TrafficLSTM(
                input_size=1,
                hidden_units=units,
                prediction_horizon=self.prediction_horizon,
                dropout=dropout
            )
        elif self.model_type == 'gru':
            self.model = TrafficGRU(
                input_size=1,
                hidden_units=units,
                prediction_horizon=self.prediction_horizon,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Total parameters: {total_params:,}")
        print("✓ Model built successfully")
    
    def train(self, epochs: int = 50, batch_size: int = 32,
              validation_split: float = 0.1, learning_rate: float = 0.0001):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
       
        print(f"\nStarting training for {epochs} epochs...")
       
        # Split validation data
        val_size = int(len(self.X_train) * validation_split)
        X_train_split = self.X_train[:-val_size]
        y_train_split = self.y_train[:-val_size]
        X_val = self.X_train[-val_size:]
        y_val = self.y_train[-val_size:]
       
        # Create data loaders
        train_dataset = TrafficDataset(X_train_split, y_train_split)
        val_dataset = TrafficDataset(X_val, y_val)
       
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
       
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
       
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 300 # Number of epochs to wait before early stopping
       
        # Training loop
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
               
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
               
                train_loss += loss.item()
           
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
           
            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                   
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()
           
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
           
            # Learning rate scheduler
            scheduler.step(val_loss)


            #Explicit Messages
            # Learning rate scheduler
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']


            # Print if learning rate changed
            if new_lr != old_lr:
                print(f"  → Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
           
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
           
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
       
        # Restore best model
        self.model.load_state_dict(self.best_model_state)
        print("✓ Training complete")

    # def train(self, epochs: int = 50, batch_size: int = 32, 
    #       validation_split: float = 0.1, learning_rate: float = 0.0001,
    #       use_mixed_precision: bool = True):
    #     """
    #     Train the model with optional Mixed Precision (FP16) for A100
    
    #     Args:
    #         use_mixed_precision: Enable FP16 training (50-60% speedup on A100)
    #     """
    #     if self.model is None:
    #         raise ValueError("Model not built. Call build_model() first.")
    
    #     print(f"\nStarting training for {epochs} epochs...")
    
    #     # Split validation data
    #     val_size = int(len(self.X_train) * validation_split)
    #     X_train_split = self.X_train[:-val_size]
    #     y_train_split = self.y_train[:-val_size]
    #     X_val = self.X_train[-val_size:]
    #     y_val = self.y_train[-val_size:]
    
    #     # Create data loaders
    #     train_dataset = TrafficDataset(X_train_split, y_train_split)
    #     val_dataset = TrafficDataset(X_val, y_val)
    
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    
    #     # Loss and optimizer
    #     criterion = nn.MSELoss()
    #     optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, mode='min', factor=0.5, patience=10
    #     )
    
    #     # Mixed Precision Setup (A100 optimization!)
    #     scaler = None
    #     if use_mixed_precision and torch.cuda.is_available():
    #         from torch.amp import GradScaler, autocast
    #         scaler = GradScaler('cuda')
    #         print("  Mixed Precision (FP16) ENABLED - expect 50-60% speedup!")
    
    #     best_val_loss = float('inf')
    #     patience_counter = 0
    #     patience = 300  # Will be overridden by config
    
    #     # Training loop
    #     for epoch in range(epochs):
    #         # Train
    #         self.model.train()
    #         train_loss = 0
        
    #         for X_batch, y_batch in train_loader:
    #             X_batch = X_batch.to(self.device, non_blocking=True)
    #             y_batch = y_batch.to(self.device, non_blocking=True)
            
    #             optimizer.zero_grad()
            
    #             if scaler:
    #                 # MIXED PRECISION TRAINING (A100 optimized!)
    #                 with autocast('cuda'):
    #                     predictions = self.model(X_batch)
    #                     loss = criterion(predictions, y_batch)
                    
    #                 # Backward with gradient scaling
    #                 scaler.scale(loss).backward()
                    
    #                 # Gradient clipping (prevent exploding gradients)
    #                 scaler.unscale_(optimizer)
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
    #                 scaler.step(optimizer)
    #                 scaler.update()
    #             else:
    #                 # Regular FP32 training
    #                 predictions = self.model(X_batch)
    #                 loss = criterion(predictions, y_batch)
    #                 loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #                 optimizer.step()
                
    #             train_loss += loss.item()
            
    #         train_loss /= len(train_loader)
    #         self.train_losses.append(train_loss)
            
    #         # Validate
    #         self.model.eval()
    #         val_loss = 0
    #         with torch.no_grad():
    #             for X_batch, y_batch in val_loader:
    #                 X_batch = X_batch.to(self.device, non_blocking=True)
    #                 y_batch = y_batch.to(self.device, non_blocking=True)
                    
    #                 if scaler:
    #                     with autocast('cuda'):
    #                         predictions = self.model(X_batch)
    #                         loss = criterion(predictions, y_batch)
    #                 else:
    #                     predictions = self.model(X_batch)
    #                     loss = criterion(predictions, y_batch)
                    
    #                 val_loss += loss.item()
            
    #         val_loss /= len(val_loader)
    #         self.val_losses.append(val_loss)
            
    #         # Learning rate scheduler
    #         old_lr = optimizer.param_groups[0]['lr']
    #         scheduler.step(val_loss)
    #         new_lr = optimizer.param_groups[0]['lr']
            
    #         # Print if learning rate changed
    #         if new_lr != old_lr:
    #             print(f"  → Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
            
    #         # Print progress
    #         if (epoch + 1) % 5 == 0:
    #             print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
    #         # Early stopping
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             patience_counter = 0
    #             # Save best model
    #             self.best_model_state = self.model.state_dict()
    #         else:
    #             patience_counter += 1
    #             if patience_counter >= patience:
    #                 print(f"\nEarly stopping at epoch {epoch+1}")
    #                 break
        
    #     # Restore best model
    #     self.model.load_state_dict(self.best_model_state)
    #     print("✓ Training complete")
    
    def evaluate(self) -> dict:
        """Evaluate model on test set"""
        print("\nEvaluating model on test set...")
        
        self.model.eval()
        test_dataset = TrafficDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                predictions = self.model(X_batch)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.numpy())
        
        y_pred = np.concatenate(all_predictions, axis=0)
        y_test = np.concatenate(all_targets, axis=0)
        
        # Denormalize
        y_test_denorm = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_denorm = self.scaler.inverse_transform(y_pred.reshape(-1, 1))
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
        rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))
        r2 = r2_score(y_test_denorm, y_pred_denorm)
        
        metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
        print(f"  MAE:  {mae:.2f} bytes")
        print(f"  RMSE: {rmse:.2f} bytes")
        print(f"  R²:   {r2:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path: str = 'training_history.png'):
        """Plot training history"""
        plt.figure(figsize=(10, 4))
        
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.title('Training History')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"✓ Training history saved to {save_path}")
        plt.close()
    
    def save_model(self, model_path: str = 'models/traffic_model.pt',
                   scaler_path: str = 'models/scaler.pkl'):
        """Save trained model and scaler"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model with configuration
        torch.save({
            'model': self.model,
            'model_state_dict': self.model.state_dict(),
            'config': {
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'num_features': 1,
                'model_type': self.model_type
            }
        }, model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved to {scaler_path}")


def main():
    """Example training script"""
    print("=" * 60)
    print("Traffic Prediction Model Training (PyTorch)")
    print("=" * 60)
    
    trainer = TrafficModelTrainer(
        sequence_length=60,
        prediction_horizon=30,
        model_type='lstm'
    )
    
    trainer.load_data('data/training/traffic_history.csv')
    trainer.build_model(units=[64, 32], dropout=0.2)
    trainer.train(epochs=50, batch_size=32)
    
    metrics = trainer.evaluate()
    trainer.plot_training_history('docs/training_history.png')
    trainer.save_model('models/traffic_model.pt', 'models/scaler.pkl')
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

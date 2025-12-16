# test_gpu.py
"""
Quick test to verify PyTorch GPU setup and basic LSTM functionality
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 60)
print("PyTorch GPU Test")
print("=" * 60)

# Check GPU
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test simple LSTM
print("\n" + "=" * 60)
print("Testing LSTM on GPU")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Create simple LSTM
class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Create model and move to GPU
model = SimpleLSTM().to(device)
print(f"\n✓ Model created and moved to {device}")

# Test forward pass
batch_size = 8
sequence_length = 60
x = torch.randn(batch_size, sequence_length, 1).to(device)

print(f"\nInput shape: {x.shape}")
print(f"Input device: {x.device}")

# Forward pass
with torch.no_grad():
    output = model(x)

print(f"\nOutput shape: {output.shape}")
print(f"Output device: {output.device}")

print("\n" + "=" * 60)
print("✓ All tests passed! Ready for training!")
print("=" * 60)

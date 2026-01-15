# inspect_scaler.py
"""
Inspect and visualize what the scaler does
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load scaler
scenario = 'ddos'  # Change to check different scenarios
scaler = joblib.load(f'models/{scenario}_ultimate_scaler.pkl')

print("=" * 70)
print(f"SCALER INSPECTION - {scenario.upper()} Model")
print("=" * 70)

# Scaler properties
print("\nScaler Properties:")
print(f"  Type: {type(scaler).__name__}")
print(f"  Feature range: {scaler.feature_range}")
print(f"  Number of features: {scaler.n_features_in_}")

# Min and Max values learned from training data
print("\nLearned from training data:")
print(f"  Data min: {scaler.data_min_[0]:.2f} bytes/s")
print(f"  Data max: {scaler.data_max_[0]:.2f} bytes/s")
print(f"  Data range: {scaler.data_range_[0]:.2f} bytes/s")

# Scale and offset used for normalization
print("\nTransformation parameters:")
print(f"  Scale factor: {scaler.scale_[0]:.6f}")
print(f"  Min offset: {scaler.min_[0]:.6f}")

# Formula
print("\nNormalization formula:")
print(f"  normalized = (original - {scaler.data_min_[0]:.1f}) / {scaler.data_range_[0]:.1f}")

# Test examples
print("\n" + "=" * 70)
print("EXAMPLE TRANSFORMATIONS")
print("=" * 70)

test_values = [
    scaler.data_min_[0],  # Minimum seen
    scaler.data_max_[0],  # Maximum seen
    (scaler.data_min_[0] + scaler.data_max_[0]) / 2,  # Middle
    scaler.data_min_[0] * 0.5,  # Below min (extrapolation)
    scaler.data_max_[0] * 1.5   # Above max (extrapolation)
]

print(f"\n{'Original Value':<20} {'Normalized':<15} {'Back to Original':<20}")
print("-" * 70)

for val in test_values:
    normalized = scaler.transform([[val]])[0][0]
    back = scaler.inverse_transform([[normalized]])[0][0]
    print(f"{val:<20.2f} {normalized:<15.4f} {back:<20.2f}")

# Visualization
print("\n\nGenerating visualization...")

# Create range of values
original_values = np.linspace(0, scaler.data_max_[0] * 1.2, 1000)
normalized_values = scaler.transform(original_values.reshape(-1, 1)).flatten()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Normalization curve
axes[0].plot(original_values, normalized_values, 'b-', linewidth=2)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=scaler.data_min_[0], color='red', linestyle='--', alpha=0.5, label='Training Min')
axes[0].axvline(x=scaler.data_max_[0], color='green', linestyle='--', alpha=0.5, label='Training Max')
axes[0].set_xlabel('Original Traffic (bytes/s)')
axes[0].set_ylabel('Normalized Value (0-1)')
axes[0].set_title(f'{scenario.upper()} - Normalization Function')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Distribution of training data
axes[1].hist(normalized_values, bins=50, alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Min (0)')
axes[1].axvline(x=1, color='green', linestyle='--', linewidth=2, label='Max (1)')
axes[1].set_xlabel('Normalized Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution After Normalization')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'docs/scaler_visualization_{scenario}.png', dpi=150)
print(f"✓ Visualization saved to docs/scaler_visualization_{scenario}.png")

plt.show()

print("\n" + "=" * 70)

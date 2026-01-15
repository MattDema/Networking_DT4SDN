# compare_all_scalers.py
"""
Compare scalers across all scenarios
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np

scenarios = ['normal', 'burst', 'congestion', 'ddos', 'mixed']
colors = ['blue', 'orange', 'green', 'red', 'purple']

print("=" * 70)
print("COMPARING ALL SCALERS")
print("=" * 70)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for scenario, color in zip(scenarios, colors):
    try:
        scaler = joblib.load(f'models/{scenario}_ultimate_scaler.pkl')
        
        print(f"\n{scenario.upper()}:")
        print(f"  Range: [{scaler.data_min_[0]:.0f}, {scaler.data_max_[0]:.0f}] bytes/s")
        
        # Plot normalization curves
        original = np.linspace(0, scaler.data_max_[0] * 1.1, 500)
        normalized = scaler.transform(original.reshape(-1, 1)).flatten()
        
        axes[0].plot(original, normalized, linewidth=2, label=scenario, color=color)
        
        # Plot data ranges as bars
        axes[1].barh(scenario, scaler.data_max_[0] - scaler.data_min_[0], 
                     left=scaler.data_min_[0], color=color, alpha=0.6)
        
    except Exception as e:
        print(f"⚠️  Could not load {scenario} scaler: {e}")

# Configure plots
axes[0].set_xlabel('Original Traffic (bytes/s)', fontsize=12)
axes[0].set_ylabel('Normalized (0-1)', fontsize=12)
axes[0].set_title('Normalization Functions Across Scenarios', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Traffic (bytes/s)', fontsize=12)
axes[1].set_ylabel('Scenario', fontsize=12)
axes[1].set_title('Data Ranges by Scenario', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('docs/all_scalers_comparison.png', dpi=150)
print("\n✓ Comparison saved to docs/all_scalers_comparison.png")
plt.show()

print("\n" + "=" * 70)

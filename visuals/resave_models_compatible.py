# resave_models_compatible.py
"""
Resave PyTorch models in a format compatible with older Python/NumPy versions.

The issue: Models saved with NumPy 2.x can't be loaded on systems with NumPy 1.x
because of 'numpy._core' vs 'numpy.core' differences.

Solution: Resave models using pickle protocol 4 (compatible with Python 3.4+)
and avoid numpy-specific serialization issues.
"""

import torch
import os
import glob

MODELS_DIR = 'models'
OUTPUT_DIR = 'models'  # Overwrite originals

def resave_model(model_path):
    """Resave a model in a compatible format."""
    print(f"\nProcessing: {model_path}")
    
    try:
        # Load with current environment
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Get the model state dict and config
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
            
            # Convert any numpy arrays in config to Python native types
            clean_config = {}
            for key, value in config.items():
                if hasattr(value, 'tolist'):  # numpy array
                    clean_config[key] = value.tolist()
                elif hasattr(value, 'item'):  # numpy scalar
                    clean_config[key] = value.item()
                elif isinstance(value, dict):
                    # Recursively clean nested dicts (like thresholds)
                    clean_config[key] = {}
                    for k, v in value.items():
                        if hasattr(v, 'tolist'):
                            clean_config[key][k] = v.tolist()
                        elif hasattr(v, 'item'):
                            clean_config[key][k] = v.item()
                        else:
                            clean_config[key][k] = v
                else:
                    clean_config[key] = value
            
            # Create new checkpoint with clean data
            new_checkpoint = {
                'model_state_dict': state_dict,
                'config': clean_config
            }
            
            # Save with pickle protocol 4 for compatibility
            output_path = model_path  # Overwrite
            torch.save(new_checkpoint, output_path, pickle_protocol=4)
            
            print(f"  ✓ Resaved: {output_path}")
            print(f"    Config: {clean_config}")
            return True
            
        else:
            print(f"  ⚠ Unknown checkpoint format, skipping")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("RESAVE MODELS FOR COMPATIBILITY")
    print("="*60)
    
    # Find all classifier models
    pattern = os.path.join(MODELS_DIR, '*_classifier_3050.pt')
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"No models found matching: {pattern}")
        return
    
    print(f"Found {len(model_files)} models to resave:")
    for f in model_files:
        print(f"  - {f}")
    
    success = 0
    for model_path in model_files:
        if resave_model(model_path):
            success += 1
    
    print("\n" + "="*60)
    print(f"DONE: {success}/{len(model_files)} models resaved successfully")
    print("="*60)
    print("\nNow commit and push, then pull in the VM:")
    print("  git add models/ && git commit -m 'Resave models for compatibility' && git push")


if __name__ == '__main__':
    main()

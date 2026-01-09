#!/usr/bin/env python3
"""
Digital Twin Project - Environment Test Script
Run this to verify your development environment is correctly set up.

Usage:
    python test_setup.py
    
    or with virtual environment:
    source venv/bin/activate  # Mac/Linux
    venv\Scripts\activate     # Windows
    python test_setup.py
"""

import sys
import platform

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_section(text):
    """Print a formatted section header."""
    print(f"\n{text}")
    print("-" * 60)

def test_python_version():
    """Check Python version compatibility."""
    print_section("Python Version Check")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Python version: {version_str}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version is compatible (3.8+)")
        return True
    else:
        print("✗ Python 3.8 or higher required")
        return False

def test_imports():
    """Test if all required packages are installed."""
    print_section("Package Import Test")
    
    packages = [
        ('numpy', 'NumPy', False),
        ('pandas', 'Pandas', False),
        ('sklearn', 'Scikit-learn', False),
        ('tensorflow', 'TensorFlow', False),
        ('flask', 'Flask', False),
        ('matplotlib', 'Matplotlib', False),
        ('requests', 'Requests', False),
        ('yaml', 'PyYAML', False),
    ]
    
    all_good = True
    
    for module, name, optional in packages:
        try:
            __import__(module)
            print(f"✓ {name:20} - OK")
        except ImportError:
            if optional:
                print(f"⚠ {name:20} - MISSING (optional)")
            else:
                print(f"✗ {name:20} - MISSING (required)")
                all_good = False
    
    return all_good

def test_file_structure():
    """Check if project structure is correct."""
    print_section("Project Structure Check")
    
    import os
    
    required_dirs = [
        'src',
        'src/controllers',
        'src/ml_models',
        'src/web_interface',
        'src/database',
        'src/utils',
        'data',
        'tests',
        'docs',
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ {dir_path:30} - exists")
        else:
            print(f"✗ {dir_path:30} - missing")
            all_good = False
    
    return all_good

def test_tensorflow():
    """Special test for TensorFlow (can be slow to import)."""
    print_section("TensorFlow Detailed Check")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Check if GPU is available (optional)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU(s) detected: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("ℹ No GPU detected (CPU will be used)")
        
        return True
    except ImportError:
        print("✗ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"⚠ TensorFlow installed but error occurred: {e}")
        return False

def main():
    """Run all tests."""
    print_header("Digital Twin for SDN - Environment Test")
    
    results = {
        'Python Version': test_python_version(),
        'Package Imports': test_imports(),
        'File Structure': test_file_structure(),
        'TensorFlow': test_tensorflow(),
    }
    
    # Summary
    print_header("Test Summary")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} {status}")
    
    print("\n" + "=" * 60)
    
    if all(results.values()):
        print("✓ All tests passed! Environment is ready.")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Make sure virtual environment is activated")
        print("2. Run: pip install -r requirements.txt")
        print("3. Check Python version is 3.8+")
        return 1

if __name__ == "__main__":
    sys.exit(main())

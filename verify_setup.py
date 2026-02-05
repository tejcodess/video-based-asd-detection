"""
Quick Setup Verification Script
This script checks if your environment is properly configured
"""

import os
import sys

def check_imports():
    """Check if all required packages are installed"""
    print("\n" + "=" * 80)
    print("CHECKING PYTHON PACKAGES")
    print("=" * 80)
    
    packages = {
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif package == 'cv2':
                import cv2
                version = cv2.__version__
            else:
                mod = __import__(package)
                version = mod.__version__
            print(f"✓ {name:20s} version {version}")
        except ImportError as e:
            print(f"✗ {name:20s} NOT INSTALLED")
            all_ok = False
        except Exception as e:
            print(f"? {name:20s} installed but version unknown")
    
    return all_ok

def check_gpu():
    """Check GPU availability"""
    print("\n" + "=" * 80)
    print("CHECKING GPU")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            print("\n⚠ No GPU detected")
            print("  Training will use CPU (slower)")
            print("\n  For GPU support, ensure:")
            print("    - NVIDIA GPU with CUDA support")
            print("    - CUDA Toolkit 11.2+ installed")
            print("    - cuDNN 8.1+ installed")
            return False
        
        print(f"\n✓ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        # Check if GPU is actually usable
        print(f"\n  CUDA Built: {tf.test.is_built_with_cuda()}")
        print(f"  GPU Available: {tf.test.is_gpu_available()}")
        
        # Try to configure GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  Memory Growth: Enabled")
        except:
            print(f"  Memory Growth: Could not enable")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error checking GPU: {e}")
        return False

def check_config():
    """Check if config.py is properly set up"""
    print("\n" + "=" * 80)
    print("CHECKING CONFIGURATION")
    print("=" * 80)
    
    try:
        from config import (DATASET_BASE, TRAINING_DATA_PATH, TESTING_DATA_PATH,
                           MODELS_PATH, REPORTS_PATH, FEATURES_PATH, CLASSES)
        
        print(f"\nConfiguration loaded successfully!")
        print(f"\n  Dataset Base: {DATASET_BASE}")
        print(f"  Training: {TRAINING_DATA_PATH}")
        print(f"  Testing: {TESTING_DATA_PATH}")
        print(f"  Models: {MODELS_PATH}")
        print(f"  Reports: {REPORTS_PATH}")
        print(f"  Features: {FEATURES_PATH}")
        print(f"  Classes: {CLASSES}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error loading config.py: {e}")
        return False

def check_dataset():
    """Check if dataset directories exist"""
    print("\n" + "=" * 80)
    print("CHECKING DATASET")
    print("=" * 80)
    
    try:
        from config import TRAINING_DATA_PATH, TESTING_DATA_PATH, CLASSES
        
        # Check training set
        print(f"\nTraining Set: {TRAINING_DATA_PATH}")
        if os.path.exists(TRAINING_DATA_PATH):
            print("  ✓ Directory exists")
            for cls in CLASSES:
                cls_path = os.path.join(TRAINING_DATA_PATH, cls)
                if os.path.exists(cls_path):
                    video_files = [f for f in os.listdir(cls_path) 
                                  if os.path.isfile(os.path.join(cls_path, f))]
                    print(f"  ✓ Class '{cls}': {len(video_files)} files")
                else:
                    print(f"  ✗ Class '{cls}': Directory not found")
        else:
            print("  ✗ Directory NOT FOUND")
        
        # Check testing set
        print(f"\nTesting Set: {TESTING_DATA_PATH}")
        if os.path.exists(TESTING_DATA_PATH):
            print("  ✓ Directory exists")
            for cls in CLASSES:
                cls_path = os.path.join(TESTING_DATA_PATH, cls)
                if os.path.exists(cls_path):
                    video_files = [f for f in os.listdir(cls_path) 
                                  if os.path.isfile(os.path.join(cls_path, f))]
                    print(f"  ✓ Class '{cls}': {len(video_files)} files")
                else:
                    print(f"  ✗ Class '{cls}': Directory not found")
        else:
            print("  ✗ Directory NOT FOUND")
        
        return True
    except Exception as e:
        print(f"\n✗ Error checking dataset: {e}")
        return False

def check_output_dirs():
    """Check if output directories are created"""
    print("\n" + "=" * 80)
    print("CHECKING OUTPUT DIRECTORIES")
    print("=" * 80)
    
    try:
        from config import MODELS_PATH, REPORTS_PATH, FEATURES_PATH
        
        dirs = {
            'Models': MODELS_PATH,
            'Reports': REPORTS_PATH,
            'Features': FEATURES_PATH
        }
        
        for name, path in dirs.items():
            if os.path.exists(path):
                print(f"  ✓ {name}: {path}")
            else:
                print(f"  ✗ {name}: {path} (will be created automatically)")
        
        return True
    except Exception as e:
        print(f"\n✗ Error checking output directories: {e}")
        return False

def main():
    print("\n" + "=" * 80)
    print("VIDEO NEURAL NETWORK ASD SCREENING - SETUP VERIFICATION")
    print("=" * 80)
    
    results = {
        'Packages': check_imports(),
        'GPU': check_gpu(),
        'Configuration': check_config(),
        'Dataset': check_dataset(),
        'Output Directories': check_output_dirs()
    }
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for check, status in results.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"  {check:20s} {status_str}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 80)
        print("✓ ALL CHECKS PASSED!")
        print("=" * 80)
        print("\nYou can now proceed with:")
        print("  1. Training: python train_simple.py")
        print("  2. Prediction: python predict_simple.py")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ SOME CHECKS FAILED")
        print("=" * 80)
        print("\nPlease fix the issues above before proceeding.")
        print("Refer to PROJECT_UPDATE_GUIDE.md for help.")
        print("=" * 80)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

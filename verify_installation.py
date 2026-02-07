"""Quick installation verification script"""

print("=== Installation Verification ===\n")

try:
    import numpy
    print(f"✓ NumPy: {numpy.__version__}")
except Exception as e:
    print(f"✗ NumPy: {e}")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU Available: {len(gpus)} device(s)")
    else:
        print("⚠ GPU: Not detected (CPU mode)")
except Exception as e:
    print(f"✗ TensorFlow: {e}")

try:
    import keras
    print(f"✓ Keras: {keras.__version__}")
except Exception as e:
    print(f"✗ Keras: {e}")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV: {e}")

try:
    import pandas
    print(f"✓ Pandas: {pandas.__version__}")
except Exception as e:
    print(f"✗ Pandas: {e}")

try:
    import sklearn
    print(f"✓ Scikit-learn: {sklearn.__version__}")
except Exception as e:
    print(f"✗ Scikit-learn: {e}")

try:
    import h5py
    print(f"✓ h5py: {h5py.__version__}")
except Exception as e:
    print(f"✗ h5py: {e}")

print("\n=== All packages installed successfully! ===")
print("You can now run:")
print("  python train_asd_model.py")
print("  python predict_asd_model.py")

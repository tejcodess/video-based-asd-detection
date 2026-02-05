"""
Quick Test Script - Fast End-to-End Testing
Trains VGG16-LSTM on mini dataset (5-15 minutes)
"""

import numpy as np
from keras import backend as K
import os
import time
import sys

# Add GPU configuration
print("="*70)
print("INITIALIZING GPU...")
print("="*70)
try:
    from gpu_utils import configure_gpu
    gpu_configured = configure_gpu(memory_growth=True)
    if gpu_configured:
        print("✓ GPU configured successfully")
    else:
        print("⚠ No GPU detected, using CPU (will be slower)")
except Exception as e:
    print(f"⚠ GPU configuration failed: {e}")
    print("Continuing with CPU...")
print("="*70)

from config import TRAINING_DATA_PATH, MODELS_PATH, REPORTS_PATH, USE_MINI_DATASET
from plot_utils import plot_and_save_history
from recurrent_networks import vgg16LSTMVideoClassifier
from UCF101_loader import load_ucf

K.set_image_data_format('channels_last')
np.random.seed(42)

def main():
    """Run quick end-to-end test"""
    
    print("\n" + "="*70)
    print("QUICK TEST - VGG16-LSTM TRAINING")
    print("="*70)
    
    # Check dataset mode
    if not USE_MINI_DATASET:
        print("\n" + "!"*70)
        print("WARNING: config.py is set to use FULL DATASET")
        print("!"*70)
        print("\nThe full dataset has 9,680 training videos.")
        print("This will take 8-15 hours and use ~100 GB for features.")
        print("\nFor quick test, please:")
        print("1. Edit config.py")
        print("2. Change: USE_MINI_DATASET = True")
        print("3. Run this script again")
        print("!"*70)
        response = input("\nContinue with full dataset anyway? (type 'yes' to continue): ")
        if response.lower() != 'yes':
            print("\nAborted. Please set USE_MINI_DATASET = True in config.py")
            sys.exit(0)
    else:
        print("✓ Using mini dataset (fast test mode)")
    
    print(f"\nTraining data: {TRAINING_DATA_PATH}")
    print(f"Models will be saved to: {MODELS_PATH}")
    print(f"Reports will be saved to: {REPORTS_PATH}")
    print("="*70)
    
    start_time = time.time()
    
    # Load dataset
    print("\n[STEP 1/5] Loading dataset...")
    print("-"*70)
    dataset_base = os.path.dirname(TRAINING_DATA_PATH)
    load_ucf(dataset_base)
    print("✓ Dataset loaded")
    
    # Create classifier
    print("\n[STEP 2/5] Creating VGG16-LSTM model...")
    print("-"*70)
    print("Architecture:")
    print("  - VGG16 (pre-trained on ImageNet)")
    print("  - LSTM layer (512 units)")
    print("  - Dense layer (512 units)")
    print("  - Output layer (2 classes: ASD, TD)")
    classifier = vgg16LSTMVideoClassifier()
    print("✓ Model created")
    
    # Train
    print("\n[STEP 3/5] Training model...")
    print("-"*70)
    print("NOTE: First run will extract VGG16 features from videos")
    print("      This is slower but features are cached for future runs")
    print("      Subsequent training will be much faster!")
    print("-"*70)
    
    history = classifier.fit(
        data_dir_path=TRAINING_DATA_PATH,
        model_dir_path=MODELS_PATH,
        vgg16_include_top=False,  # Hi-dim features (25088-dim)
        data_set_name='training_set'
    )
    print("✓ Training completed")
    
    # Save plot
    print("\n[STEP 4/5] Saving training history plot...")
    print("-"*70)
    plot_file = os.path.join(REPORTS_PATH, 'quick-test-history.png')
    plot_and_save_history(history, 'quick-test', plot_file)
    print(f"✓ Plot saved: {plot_file}")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n[STEP 5/5] Test complete!")
    print("-"*70)
    
    print("\n" + "="*70)
    print("QUICK TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\nModel saved in: {MODELS_PATH}")
    print(f"Report saved in: {REPORTS_PATH}")
    
    # Show files created
    print("\n" + "="*70)
    print("FILES CREATED:")
    print("="*70)
    model_files = [
        "vgg16-lstm-hi-dim-config.npy",
        "vgg16-lstm-hi-dim-weights.h5",
        "vgg16-lstm-hi-dim-architecture.json"
    ]
    for fname in model_files:
        fpath = os.path.join(MODELS_PATH, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024*1024)
            print(f"  ✓ {fname} ({size_mb:.1f} MB)")
    
    print(f"\n  ✓ quick-test-history.png")
    
    # Next steps
    if USE_MINI_DATASET:
        print("\n" + "="*70)
        print("NEXT STEPS FOR FULL TRAINING:")
        print("="*70)
        print("1. Edit config.py and change:")
        print("   USE_MINI_DATASET = False")
        print("\n2. Run full training:")
        print("   python train_simple.py")
        print("\n3. Expected:")
        print("   - Time: 8-15 hours on GPU")
        print("   - Feature cache: ~100 GB")
        print("   - Model size: ~500 MB")
        print("\n4. Monitor training:")
        print("   - Watch GPU usage: nvidia-smi -l 1")
        print("   - Check plots in reports/autism_data/")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("FULL TRAINING COMPLETED!")
        print("="*70)
        print("You can now:")
        print("1. Run predictions: python predict_simple.py")
        print("2. Check accuracy plots in reports/autism_data/")
        print("3. Use the model for inference")
        print("="*70)
    
    print("\n✓ All done!\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("You can resume by running this script again.")
        print("Cached features will be reused.")
        sys.exit(0)
    except Exception as e:
        print("\n" + "!"*70)
        print("ERROR OCCURRED:")
        print("!"*70)
        print(f"{e}")
        print("\nTroubleshooting:")
        print("1. Check if dataset exists")
        print("2. Check if enough disk space (~2 GB for mini, ~100 GB for full)")
        print("3. Try running with USE_MINI_DATASET = True first")
        print("!"*70)
        raise

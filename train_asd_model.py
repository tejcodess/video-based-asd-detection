"""
Updated Training Script for ASD Video Classification
Adapted for Windows environment with Python 3.9+ compatibility
Dataset: D:/projects/01/dataset/autism_data_anonymized
"""

import numpy as np
import os
import sys
from pathlib import Path

# Set image format for Keras backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from plot_utils import plot_and_save_history
from recurrent_networks import vgg16LSTMVideoClassifier

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SETUP
# ============================================================================

# Your dataset location
DATASET_PATH = r"D:\projects\01\dataset\autism_data_anonymized\training_set"
DATA_SET_NAME = 'autism_data'

# Output directories (relative to script location)
PROJECT_ROOT = Path(__file__).parent
OUTPUT_MODEL_DIR = PROJECT_ROOT / 'models' / DATA_SET_NAME
REPORT_DIR = PROJECT_ROOT / 'reports' / DATA_SET_NAME

# Model configuration
VGG16_INCLUDE_TOP = False  # Use high-dimensional features (recommended)
TEST_SIZE = 0.2  # 20% of training data for validation
RANDOM_STATE = 42

# Training parameters (optimized for 80 training videos)
# The batch size will be calculated automatically based on dataset size
NUM_EPOCHS = 100  # Adjust based on your needs (start with 50-100)

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def check_gpu_availability():
    """Check if GPU is available for training"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU available: {len(gpus)} device(s) detected")
        for gpu in gpus:
            print(f"  - {gpu}")
        return True
    else:
        print("⚠ No GPU detected. Training will use CPU (slower)")
        return False


def verify_dataset_structure():
    """Verify the dataset exists and has correct structure"""
    dataset_path = Path(DATASET_PATH)
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}\n"
            f"Please update DATASET_PATH in the script or ensure the dataset is at the correct location."
        )
    
    print(f"\n{'='*70}")
    print(f"Dataset Verification")
    print(f"{'='*70}")
    print(f"Dataset location: {dataset_path}")
    
    # Check for subdirectories (classes)
    classes = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(classes) == 0:
        raise ValueError(f"No class folders found in {dataset_path}")
    
    print(f"\nFound {len(classes)} classes:")
    total_videos = 0
    
    for class_dir in sorted(classes):
        video_files = list(class_dir.glob('*.avi')) + list(class_dir.glob('*.mp4'))
        video_count = len(video_files)
        total_videos += video_count
        print(f"  - {class_dir.name}: {video_count} videos")
    
    print(f"\nTotal training videos: {total_videos}")
    
    if total_videos == 0:
        raise ValueError(f"No video files (.avi or .mp4) found in class folders")
    
    print(f"{'='*70}\n")
    return True


def create_output_directories():
    """Create necessary output directories"""
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directories ready:")
    print(f"  - Models: {OUTPUT_MODEL_DIR}")
    print(f"  - Reports: {REPORT_DIR}")


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("ASD Video Classification - Training Script")
    print("="*70)
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print("="*70 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    
    # Step 1: Check GPU availability
    check_gpu_availability()
    
    # Step 2: Verify dataset structure
    verify_dataset_structure()
    
    # Step 3: Create output directories
    create_output_directories()
    
    # Step 4: Initialize the classifier
    print("\n" + "="*70)
    print("Initializing VGG16-LSTM Classifier")
    print("="*70)
    print(f"VGG16 Include Top: {VGG16_INCLUDE_TOP}")
    print(f"Using high-dimensional features: {not VGG16_INCLUDE_TOP}")
    print("="*70 + "\n")
    
    classifier = vgg16LSTMVideoClassifier()
    
    # Step 5: Train the model
    print("\n" + "="*70)
    print("Starting Training Process")
    print("="*70)
    print(f"This will:")
    print(f"1. Extract VGG16 features from videos (one-time process)")
    print(f"2. Train LSTM model on extracted features")
    print(f"3. Save the best model to: {OUTPUT_MODEL_DIR}")
    print(f"4. Generate training history plots in: {REPORT_DIR}")
    print("="*70 + "\n")
    
    try:
        history = classifier.fit(
            data_dir_path=str(DATASET_PATH),
            model_dir_path=str(OUTPUT_MODEL_DIR),
            vgg16_include_top=VGG16_INCLUDE_TOP,
            data_set_name=DATA_SET_NAME,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        
        # Step 6: Save training history plot
        print("\n" + "="*70)
        print("Saving Training History")
        print("="*70)
        
        model_name = vgg16LSTMVideoClassifier.model_name
        if VGG16_INCLUDE_TOP:
            history_file = REPORT_DIR / f'{model_name}-history.png'
        else:
            history_file = REPORT_DIR / f'{model_name}-hi-dim-history.png'
        
        plot_and_save_history(history, model_name, str(history_file))
        print(f"✓ Training history saved to: {history_file}")
        print("="*70 + "\n")
        
        # Step 7: Display training results
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        
        print(f"\nModel files saved in: {OUTPUT_MODEL_DIR}")
        print(f"  - Config: {model_name}-hi-dim-config.npy")
        print(f"  - Weights: {model_name}-hi-dim-weights.h5")
        print(f"  - Architecture: {model_name}-hi-dim-architecture.json")
        
        print(f"\nNext steps:")
        print(f"1. Review training history plot at: {history_file}")
        print(f"2. Use 'predict_asd_model.py' to run predictions on test data")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: Training failed")
        print(f"{'='*70}")
        print(f"Error message: {str(e)}")
        print(f"\nTroubleshooting tips:")
        print(f"1. Verify dataset path: {DATASET_PATH}")
        print(f"2. Ensure videos are in .avi or .mp4 format")
        print(f"3. Check that you have write permissions for output directories")
        print(f"4. Verify TensorFlow and Keras are properly installed")
        print(f"{'='*70}\n")
        raise


if __name__ == '__main__':
    main()

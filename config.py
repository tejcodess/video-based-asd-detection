"""
Configuration file for ASD Detection Model
Edit this file to customize paths and training parameters
"""

import os

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Base directory (change this to your dataset location)
DATASET_BASE = r"D:\projects\01\dataset\autism_data_anonymized"

# Automatically set paths
TRAINING_DATA_PATH = os.path.join(DATASET_BASE, "training_set")
TESTING_DATA_PATH = os.path.join(DATASET_BASE, "testing_set")

# Class names (folders inside training_set and testing_set)
CLASSES = ["ASD", "TD"]

# ============================================================================
# PREPROCESSING CONFIGURATION (New Section)
# ============================================================================
# Path where normalized skeletal videos will be stored
NORMALIZED_VIDEOS_PATH = os.path.join(DATASET_BASE, "normalized_videos")

# Set to True to enable raw-to-pose conversion
ENABLE_POSE_NORMALIZATION = True 

# ============================================================================
# OUTPUT PATHS
# ============================================================================

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Model outputs
MODELS_PATH = os.path.join(PROJECT_ROOT, "models", "autism_data")
CHECKPOINTS_PATH = os.path.join(MODELS_PATH, "checkpoints")

# Reports and logs
REPORTS_PATH = os.path.join(PROJECT_ROOT, "reports", "autism_data")
LOGS_PATH = os.path.join(PROJECT_ROOT, "logs")

# Feature cache (auto-generated)
FEATURES_PATH = os.path.join(PROJECT_ROOT, "extracted_features")

# Create directories if they don't exist
# Added NORMALIZED_VIDEOS_PATH to the list
for path in [MODELS_PATH, CHECKPOINTS_PATH, REPORTS_PATH, LOGS_PATH, FEATURES_PATH, NORMALIZED_VIDEOS_PATH]:
    os.makedirs(path, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

VGG16_INCLUDE_TOP = False  
IMAGE_SIZE = (224, 224)    
FRAME_RATE = 1             

# LSTM Model
LSTM_UNITS = 512
DROPOUT_RATE = 0.5
DENSE_UNITS = 512

# Training
BATCH_SIZE = 625           
EPOCHS = 100
VALIDATION_SPLIT = 0.2     
RANDOM_SEED = 42

# GPU CONFIGURATION
GPU_MEMORY_GROWTH = True
GPU_MEMORY_LIMIT = None  

def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Dataset:        {DATASET_BASE}")
    print(f"Normalization:  {'ENABLED' if ENABLE_POSE_NORMALIZATION else 'DISABLED'}")
    print(f"VGG16 top:      {VGG16_INCLUDE_TOP}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Epochs:         {EPOCHS}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()
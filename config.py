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

# Verify dataset exists
if not os.path.exists(DATASET_BASE):
    print(f"⚠️  WARNING: Dataset not found at {DATASET_BASE}")
    print(f"   Please update DATASET_BASE in config.py")

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
for path in [MODELS_PATH, CHECKPOINTS_PATH, REPORTS_PATH, LOGS_PATH, FEATURES_PATH]:
    os.makedirs(path, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# VGG16 Feature Extraction
VGG16_INCLUDE_TOP = False  # False = Hi-dim features (25088), True = Standard (1000)
IMAGE_SIZE = (224, 224)    # VGG16 input size
FRAME_RATE = 1             # Extract 1 frame per second

# LSTM Model
LSTM_UNITS = 512
DROPOUT_RATE = 0.5
DENSE_UNITS = 512

# Training
BATCH_SIZE = 625           # Reduce if GPU out of memory (try 256 or 128)
EPOCHS = 100
VALIDATION_SPLIT = 0.2     # 20% for validation
RANDOM_SEED = 42

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

# Enable GPU memory growth (prevents OOM errors)
GPU_MEMORY_GROWTH = True

# Limit GPU memory (None = use all available)
GPU_MEMORY_LIMIT = None  # Set to 4096 for 4GB limit

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Dataset:        {DATASET_BASE}")
    print(f"Training data:  {TRAINING_DATA_PATH}")
    print(f"Testing data:   {TESTING_DATA_PATH}")
    print(f"Classes:        {CLASSES}")
    print(f"Models:         {MODELS_PATH}")
    print(f"Reports:        {REPORTS_PATH}")
    print(f"")
    print(f"VGG16 top:      {VGG16_INCLUDE_TOP}")
    print(f"LSTM units:     {LSTM_UNITS}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Epochs:         {EPOCHS}")
    print(f"Validation:     {VALIDATION_SPLIT * 100}%")
    print("=" * 70)

if __name__ == "__main__":
    print_config()

import os

# ============================================================================
# DATASET MODE SELECTION
# ============================================================================
# Set to True for quick testing with mini dataset (10 videos)
# Set to False for full training (9,680 videos)
USE_MINI_DATASET = True  # <<< CHANGE THIS TO False FOR FULL TRAINING

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if USE_MINI_DATASET:
    print("=" * 70)
    print("MODE: MINI DATASET (Quick Test)")
    print("=" * 70)
    DATASET_BASE = os.path.join(BASE_DIR, "mini_dataset")
else:
    print("=" * 70)
    print("MODE: FULL DATASET (Complete Training)")
    print("=" * 70)
    DATASET_BASE = r"D:\projects\01\dataset\autism_data_anonymized"

# Data paths
TRAINING_DATA_PATH = os.path.join(DATASET_BASE, "training_set")
TESTING_DATA_PATH = os.path.join(DATASET_BASE, "testing_set")

# Output paths
MODELS_PATH = os.path.join(BASE_DIR, "models", "autism_data")
REPORTS_PATH = os.path.join(BASE_DIR, "reports", "autism_data")
FEATURES_PATH = os.path.join(BASE_DIR, "extracted_features")

# Class names
CLASSES = ["ASD", "TD"]

# Create directories if they don't exist
for path in [MODELS_PATH, REPORTS_PATH, FEATURES_PATH]:
    os.makedirs(path, exist_ok=True)

print(f"Training data: {TRAINING_DATA_PATH}")
print(f"Testing data:  {TESTING_DATA_PATH}")
print(f"Models:        {MODELS_PATH}")
print("=" * 70)

# Print configuration for verification
if __name__ == "__main__":
    print("Configuration Paths:")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  DATASET_BASE: {DATASET_BASE}")
    print(f"  TRAINING_DATA_PATH: {TRAINING_DATA_PATH}")
    print(f"  TESTING_DATA_PATH: {TESTING_DATA_PATH}")
    print(f"  MODELS_PATH: {MODELS_PATH}")
    print(f"  REPORTS_PATH: {REPORTS_PATH}")
    print(f"  FEATURES_PATH: {FEATURES_PATH}")
    print(f"  CLASSES: {CLASSES}")

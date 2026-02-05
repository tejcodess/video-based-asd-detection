# Project Update Summary - Autism Dataset Integration

## Overview
This document summarizes all changes made to adapt the Video Neural Network ASD Screening project to work with your custom dataset structure and resolve compatibility issues.

---

## Changes Implemented

### 1. Updated requirements.txt
**File:** `requirements.txt`

Changed package versions for compatibility:
- tensorflow: 2.18.0 → 2.10.0
- keras: Added as standalone 2.10.0
- numpy: 1.26.0+ → 1.23.5 (fixed version)
- opencv-contrib-python: 4.5.5.64 (specific version)

### 2. Fixed Import Statements
**Files Modified:** 
- `recurrent_networks.py`
- `vgg16_feature_extractor.py`
- `convolutional.py`
- `device_utils.py`

**Changes:**
- Changed all `from tensorflow.keras` → `from keras`
- Updated `device_utils.py` for TensorFlow 2.x compatibility
- Removed deprecated TF 1.x session configuration

### 3. Created config.py
**New File:** `config.py`

Centralized configuration with paths to your dataset:
```
Dataset Base: D:\projects\01\dataset\autism_data_anonymized
Training: D:\projects\01\dataset\autism_data_anonymized\training_set
Testing: D:\projects\01\dataset\autism_data_anonymized\testing_set
Models: <project>\models\autism_data
Reports: <project>\reports\autism_data
Features: <project>\extracted_features
Classes: ["ASD", "TD"]
```

### 4. Updated Training Script
**File:** `vgg16_lstm_hi_dim_train.py`

- Now uses paths from `config.py`
- Added print statements showing configuration
- Uses `TRAINING_DATA_PATH` directly
- Improved output messages

### 5. Updated Prediction Script
**File:** `vgg16_lstm_hi_dim_predict.py`

- Now uses paths from `config.py`
- Uses `TESTING_DATA_PATH` for predictions
- Added print statements for verification
- Shows final accuracy summary

### 6. Updated Data Loader
**File:** `UCF101_loader.py`

Modified to handle both old and new dataset structures:
- Detects `training_set`/`testing_set` folders automatically
- Works directly with `ASD` and `TD` class folders
- Added verbose output showing what's being scanned
- Improved error handling

### 7. Created train_simple.py
**New File:** `train_simple.py`

User-friendly training script with:
- Clear configuration display
- Dataset verification
- Class and video count display
- Progress messages
- Error checking
- Success confirmation

### 8. Created predict_simple.py
**New File:** `predict_simple.py`

User-friendly prediction script with:
- Configuration verification
- Model file existence checks
- Per-video prediction display with ✓/✗ status
- Running accuracy updates
- Per-class performance metrics
- Results saved to file
- Comprehensive summary

---

## Your Dataset Structure

The code now expects this structure:

```
D:\projects\01\dataset\autism_data_anonymized\
├── training_set\
│   ├── ASD\
│   │   ├── video1.mp4
│   │   ├── video2.avi
│   │   └── ...
│   └── TD\
│       ├── video1.mp4
│       ├── video2.avi
│       └── ...
└── testing_set\
    ├── ASD\
    │   └── [video files]
    └── TD\
        └── [video files]
```

Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`

---

## How to Use

### Step 1: Install Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Verify Configuration

```powershell
# Check that paths are correct
python config.py
```

Expected output:
```
Configuration Paths:
  BASE_DIR: D:\projects\01\Video-Neural-Network-ASD-screening
  DATASET_BASE: D:\projects\01\dataset\autism_data_anonymized
  TRAINING_DATA_PATH: D:\projects\01\dataset\autism_data_anonymized\training_set
  TESTING_DATA_PATH: D:\projects\01\dataset\autism_data_anonymized\testing_set
  MODELS_PATH: D:\projects\01\Video-Neural-Network-ASD-screening\models\autism_data
  REPORTS_PATH: D:\projects\01\Video-Neural-Network-ASD-screening\reports\autism_data
  FEATURES_PATH: D:\projects\01\Video-Neural-Network-ASD-screening\extracted_features
  CLASSES: ['ASD', 'TD']
```

### Step 3: Train the Model

```powershell
# Simple training (recommended)
python train_simple.py

# Or use the original script
python vgg16_lstm_hi_dim_train.py
```

What happens during training:
1. Scans training_set/ASD and training_set/TD folders
2. Extracts VGG16 features from videos (cached for reuse)
3. Trains LSTM classifier
4. Saves model to `models/autism_data/`
5. Saves training history plot to `reports/autism_data/`

### Step 4: Test the Model

```powershell
# Simple prediction (recommended)
python predict_simple.py

# Or use the original script
python vgg16_lstm_hi_dim_predict.py
```

What happens during prediction:
1. Loads trained model from `models/autism_data/`
2. Scans testing_set/ASD and testing_set/TD folders
3. Makes predictions on each video
4. Displays per-video results with ✓/✗
5. Shows overall and per-class accuracy
6. Saves results to `models/autism_data/prediction_results.txt`

---

## Key Features

### Verbose Output
All scripts now print:
- Path configurations being used
- Number of classes and videos found
- Progress during processing
- Final results and statistics

### Error Checking
Scripts verify:
- Dataset directories exist
- Video files are present
- Model files exist (for prediction)
- Classes match expected structure

### Flexibility
The code supports:
- Both old (very_large_data/autism_data) and new (training_set/testing_set) structures
- Multiple video formats
- Any two-class classification (just update CLASSES in config.py)

---

## File Reference

### Configuration
- `config.py` - Central configuration file

### Training Scripts
- `train_simple.py` - User-friendly training (recommended)
- `vgg16_lstm_hi_dim_train.py` - Original training script (updated)
- `vgg16_bidirectional_lstm_hi_dim_train.py` - Bidirectional LSTM variant
- `cnn_train.py` - CNN-only variant

### Prediction Scripts
- `predict_simple.py` - User-friendly prediction (recommended)
- `vgg16_lstm_hi_dim_predict.py` - Original prediction script (updated)

### Core Modules
- `recurrent_networks.py` - LSTM classifier implementation
- `vgg16_feature_extractor.py` - VGG16 feature extraction
- `UCF101_loader.py` - Dataset loading utilities
- `plot_utils.py` - Training visualization

---

## Troubleshooting

### Issue: "Directory does not exist"
**Solution:** Verify your dataset is at:
`D:\projects\01\dataset\autism_data_anonymized\training_set`

### Issue: "No videos found"
**Solution:** 
- Ensure videos are in `training_set/ASD` and `training_set/TD` folders
- Check video file extensions (.mp4, .avi, .mov, .mkv)

### Issue: "Model config file not found"
**Solution:** Train the model first using `train_simple.py`

### Issue: Import errors
**Solution:** Reinstall requirements:
```powershell
pip install -r requirements.txt --force-reinstall
```

### Issue: CUDA/GPU errors
**Solution:** The code will fall back to CPU automatically. For GPU support, ensure:
- CUDA toolkit installed
- tensorflow-gpu properly configured
- GPU drivers up to date

---

## Expected Training Time

- **Feature extraction:** ~1-2 seconds per video (one-time, cached)
- **Training:** Depends on dataset size
  - Small (50-100 videos): 5-15 minutes
  - Medium (100-500 videos): 15-60 minutes
  - Large (500+ videos): 1-3 hours

Features are cached as `.npy` files, so subsequent training runs are much faster.

---

## Output Files

### Models Directory
```
models/autism_data/
├── vgg16-lstm-hi-dim-config.npy
├── vgg16-lstm-hi-dim-weights.h5
├── vgg16-lstm-hi-dim-architecture.json
└── prediction_results.txt
```

### Reports Directory
```
reports/autism_data/
└── vgg16-lstm-history.png
```

### Extracted Features
```
extracted_features/
└── training_set-vgg16-HiDimFeatures/
    ├── ASD/
    │   ├── video1.npy
    │   └── ...
    └── TD/
        ├── video1.npy
        └── ...
```

---

## Next Steps

1. **Verify dataset location:** Ensure videos are in the correct folders
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Test configuration:** `python config.py`
4. **Train model:** `python train_simple.py`
5. **Test model:** `python predict_simple.py`

---

## Additional Notes

- The model uses VGG16 for feature extraction and LSTM for sequence classification
- Videos are processed at 1 frame per second
- Features are 25088-dimensional (VGG16 without top layer)
- Default: 100 epochs, batch size 625
- Random seed fixed at 42 for reproducibility

For questions or issues, refer to the original README.md or the research papers cited in the project.

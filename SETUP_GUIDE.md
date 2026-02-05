# ASD Video Classification - Setup and Usage Guide

## 🎯 Quick Start Guide for Your Local Setup

This guide provides step-by-step instructions to train and test the VGG16-LSTM neural network for Autism Spectrum Disorder (ASD) classification using your custom dataset.

### Your Dataset Configuration
- **Location**: `D:\projects\01\dataset\autism_data_anonymized`
- **Structure**:
  ```
  autism_data_anonymized/
  ├── training/
  │   ├── ASD/  (40 videos)
  │   └── TD/   (40 videos)
  └── testing/
      ├── ASD/  (40 videos)
      └── TD/   (40 videos)
  ```

---

## 📋 Prerequisites

### System Requirements
- **OS**: Windows 10/11
- **Python**: 3.9 or higher
- **GPU**: CUDA-compatible GPU recommended (but not required)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~5GB for features cache

### Python Environment

1. **Create Virtual Environment** (if not already done):
   ```powershell
   python -m venv venv
   ```

2. **Activate Virtual Environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

### Verify Installation

Run the verification script to check everything is properly set up:
```powershell
python verify_dataset.py
```

This will:
- ✅ Check Python version and installed packages
- ✅ Verify dataset structure
- ✅ Count videos in each class
- ✅ Test video file readability
- ✅ Provide recommendations

---

## 🚀 Training the Model

### Step 1: Configure Paths

Open [`train_asd_model.py`](train_asd_model.py) and verify these settings:

```python
# Your dataset location (already configured for your setup)
DATASET_PATH = r"D:\projects\01\dataset\autism_data_anonymized\training"

# Model configuration
VGG16_INCLUDE_TOP = False  # Use high-dimensional features (RECOMMENDED)
TEST_SIZE = 0.2           # 20% validation split
NUM_EPOCHS = 100          # Start with 100 epochs
```

### Step 2: Run Training

```powershell
python train_asd_model.py
```

### What Happens During Training:

1. **Feature Extraction** (First time only - takes ~30-60 minutes):
   - Extracts VGG16 features from all training videos
   - Cached in `D:\projects\01\dataset\autism_data-vgg16-HiDimFeatures\`
   - Subsequent training runs will reuse these features

2. **Model Training** (~20-60 minutes depending on GPU):
   - Trains LSTM on extracted features
   - 80 videos total: 64 for training, 16 for validation (80/20 split)
   - Saves best model based on validation accuracy

3. **Output Files**:
   - Model files saved to: `models/autism_data/`
     - `vgg16-lstm-hi-dim-config.npy` - Model configuration
     - `vgg16-lstm-hi-dim-weights.h5` - Trained weights
     - `vgg16-lstm-hi-dim-architecture.json` - Model architecture
   - Training history plot: `reports/autism_data/vgg16-lstm-hi-dim-history.png`

### Expected Training Time:
- **With GPU**: 30-90 minutes total (20-60 min feature extraction + 10-30 min training)
- **CPU Only**: 2-4 hours total (1-2 hours feature extraction + 1-2 hours training)

### Monitoring Training:

Watch the console output for:
```
Epoch 1/100
1/1 [==============================] - 5s 5s/step - loss: 0.6931 - accuracy: 0.5000 - val_loss: 0.6931 - val_accuracy: 0.5000
Epoch 2/100
1/1 [==============================] - 2s 2s/step - loss: 0.6920 - accuracy: 0.5156 - val_loss: 0.6918 - val_accuracy: 0.5000
...
```

**Good signs**:
- Loss decreasing
- Training accuracy increasing
- Validation accuracy improving (may fluctuate)

**Warning signs**:
- Validation accuracy stuck at 0.5 (random guessing)
- Large gap between training and validation accuracy (overfitting)

---

## 🔍 Testing the Model

### Step 1: Configure Test Paths

Open [`predict_asd_model.py`](predict_asd_model.py) and verify:

```python
# Your test dataset location (already configured)
TEST_DATASET_PATH = r"D:\projects\01\dataset\autism_data_anonymized\testing"

# Model configuration (must match training)
VGG16_INCLUDE_TOP = False
```

### Step 2: Run Predictions

```powershell
python predict_asd_model.py
```

### What Happens During Prediction:

1. **Model Loading**: Loads trained model from `models/autism_data/`
2. **Feature Extraction**: Extracts VGG16 features from test videos
3. **Prediction**: Classifies each video as ASD or TD
4. **Results**: Saves predictions to CSV file

### Output:

**Console Output**:
```
[1/80] Processing: video_001.avi
  True: ASD | Predicted: ASD | Confidence: 0.8523 | ✓
  Running accuracy: 1.0000 (1/1)

[2/80] Processing: video_002.avi
  True: TD | Predicted: TD | Confidence: 0.7234 | ✓
  Running accuracy: 1.0000 (2/2)
...

Overall Accuracy: 0.6875 (55/80)

Per-Class Results:
  ASD:
    Accuracy: 0.7000 (28/40)
  TD:
    Accuracy: 0.6750 (27/40)
```

**CSV Output** (`predictions_YYYYMMDD_HHMMSS.csv`):
```csv
Video Name,True Label,Predicted Label,Correct,Confidence,Class ASD Prob,Class TD Prob
video_001.avi,ASD,ASD,True,0.8523,0.8523,0.1477
video_002.avi,TD,TD,True,0.7234,0.2766,0.7234
...
```

---

## 📊 Model Performance Guidelines

### Expected Results (Based on Original Research):
- **Target Accuracy**: ~68-75% on test set
- **Baseline**: 50% (random guessing for 2 classes)

### Your Dataset (80 training videos):
- **Good performance**: >65% accuracy
- **Acceptable**: 60-65% accuracy
- **Needs improvement**: <60% accuracy

### If Accuracy is Low (<60%):

1. **Try More Epochs**:
   ```python
   NUM_EPOCHS = 150  # or 200
   ```

2. **Check for Overfitting**:
   - If training accuracy >> validation accuracy:
     - Add more dropout
     - Get more training data
     - Try data augmentation

3. **Try Bidirectional LSTM**:
   - Change classifier in `train_asd_model.py`:
     ```python
     from recurrent_networks import vgg16BidirectionalLSTMVideoClassifier
     classifier = vgg16BidirectionalLSTMVideoClassifier()
     ```

4. **Check Data Quality**:
   - Ensure videos are properly processed with OpenPose
   - Verify class labels are correct
   - Check for corrupted videos

---

## 🛠️ Troubleshooting

### Issue: "Dataset not found"
**Solution**: Update paths in training/prediction scripts:
```python
DATASET_PATH = r"YOUR\ACTUAL\PATH\TO\training"
TEST_DATASET_PATH = r"YOUR\ACTUAL\PATH\TO\testing"
```

### Issue: "Cannot open video file"
**Possible causes**:
- Corrupted video file
- Unsupported codec
- Missing opencv

**Solution**:
```powershell
pip install --upgrade opencv-python opencv-contrib-python
```

### Issue: "Out of memory" during training
**Solutions**:
1. Reduce batch size in `recurrent_networks.py`:
   ```python
   BATCH_SIZE = 32  # Instead of 625
   ```

2. Close other applications

3. Use CPU instead (slower but uses less memory):
   ```python
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
   ```

### Issue: Training is very slow
**Solutions**:
1. **Check GPU usage**:
   ```powershell
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

2. **If no GPU detected**, install CUDA toolkit and cuDNN (optional)

3. **Use CPU**: Training will take longer but will work

### Issue: "fit_generator deprecated" warning
- Already fixed in updated code
- Ignore if you see it; model will still work

### Issue: Low accuracy even after many epochs
**Checklist**:
- [ ] Dataset labels are correct
- [ ] Videos are properly preprocessed (OpenPose normalized)
- [ ] Training/testing split is balanced
- [ ] Model is not overfitting (check validation accuracy)
- [ ] Try different hyperparameters (epochs, learning rate)

---

## 📁 File Structure

```
Video-Neural-Network-ASD-screening/
├── train_asd_model.py          # ⭐ Main training script (USE THIS)
├── predict_asd_model.py         # ⭐ Main prediction script (USE THIS)
├── verify_dataset.py            # ⭐ Dataset verification tool
├── recurrent_networks.py        # Model architecture (updated for Python 3.9+)
├── vgg16_feature_extractor.py  # Feature extraction (updated for Python 3.9+)
├── asd_data_loader.py          # Custom data loader
├── plot_utils.py               # Training history visualization
├── requirements.txt            # Python dependencies
├── models/                     # Trained models saved here
│   └── autism_data/
│       ├── vgg16-lstm-hi-dim-config.npy
│       ├── vgg16-lstm-hi-dim-weights.h5
│       └── vgg16-lstm-hi-dim-architecture.json
├── reports/                    # Training plots saved here
│   └── autism_data/
│       └── vgg16-lstm-hi-dim-history.png
└── predictions_*.csv           # Prediction results

# Your dataset location (external):
D:/projects/01/dataset/autism_data_anonymized/
├── training/
│   ├── ASD/
│   └── TD/
└── testing/
    ├── ASD/
    └── TD/

# Feature cache (created automatically):
D:/projects/01/dataset/autism_data-vgg16-HiDimFeatures/
├── training/
│   ├── ASD/
│   └── TD/
```

---

## 🔬 Advanced Usage

### Custom Hyperparameters

Edit `recurrent_networks.py` to adjust:

```python
BATCH_SIZE = 625      # Adjust based on dataset size
NUM_EPOCHS = 100      # More epochs for better convergence
HIDDEN_UNITS = 512    # LSTM hidden units
```

### Model Variants

The code includes several model architectures:

1. **vgg16LSTMVideoClassifier** (Recommended):
   - Single LSTM layer
   - Good for smaller datasets
   - Faster training

2. **vgg16BidirectionalLSTMVideoClassifier**:
   - Bidirectional LSTM (processes sequences forward and backward)
   - Better accuracy but slower
   - More parameters to train

### Prediction Aggregation

The prediction CSV can be used for advanced analysis:

```python
import pandas as pd

# Load predictions
df = pd.read_csv('predictions_YYYYMMDD_HHMMSS.csv')

# High-confidence predictions only (>90%)
high_conf = df[df['Confidence'].astype(float) > 0.90]
accuracy_high_conf = high_conf['Correct'].sum() / len(high_conf)
print(f"Accuracy (>90% confidence): {accuracy_high_conf:.2%}")

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(df['True Label'], df['Predicted Label'])
print("Confusion Matrix:")
print(cm)
```

---

## 📚 Additional Resources

### Original Research Papers:
1. [Scientific Reports (2021)](https://www.nature.com/articles/s41598-021-94378-z)
2. [PLOS ONE (2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308388)

### Dataset:
- [Zenodo: OpenPose ADOS Videos](https://zenodo.org/records/12658214)

### Related Projects:
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [Original UCF-101 Classifier](https://github.com/jibinmathew69/VideoClassifier-CNNLSTM)

---

## ✅ Checklist for Success

Before training:
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset structure verified (`python verify_dataset.py`)
- [ ] Paths configured in `train_asd_model.py`
- [ ] Sufficient disk space for feature cache (~2-5GB)

After training:
- [ ] Training completed without errors
- [ ] Model files saved in `models/autism_data/`
- [ ] Training history plot generated
- [ ] Validation accuracy > 0.5 (better than random)

Before prediction:
- [ ] Paths configured in `predict_asd_model.py`
- [ ] Test dataset structure matches training dataset
- [ ] Model files exist in `models/autism_data/`

After prediction:
- [ ] Predictions CSV generated
- [ ] Overall accuracy calculated
- [ ] Per-class results reviewed

---

## 🆘 Getting Help

If you encounter issues not covered in this guide:

1. Check the console output for specific error messages
2. Verify dataset structure with `python verify_dataset.py`
3. Ensure all dependencies are properly installed
4. Check GPU availability if training is very slow
5. Review the original repository: [https://github.com/AutismBrainBehavior/Video-Neural-Network-ASD-screening](https://github.com/AutismBrainBehavior/Video-Neural-Network-ASD-screening)

---

## 📝 Summary of Changes from Original Repo

### Updates for Your Setup:
1. ✅ **Python 3.9+ Compatibility**:
   - Updated Keras imports to `tensorflow.keras`
   - Replaced deprecated `fit_generator` with `fit`
   - Fixed `np_utils.to_categorical` to `to_categorical`

2. ✅ **Windows Path Support**:
   - Uses `pathlib.Path` for cross-platform compatibility
   - Raw string literals for Windows paths

3. ✅ **Custom Dataset Structure**:
   - Works with `training/` and `testing/` subdirectories
   - No need for `very_large_data/autism_data/` structure
   - Automatic feature caching in dataset parent directory

4. ✅ **Improved User Experience**:
   - Clear console output with progress indicators
   - Detailed error messages
   - CSV output for predictions
   - Automated directory creation

5. ✅ **Better Error Handling**:
   - Validates dataset structure before training
   - Checks for corrupted video files
   - Handles missing model files gracefully

---

**Ready to start? Run the verification script first:**
```powershell
python verify_dataset.py
```

**Then train your model:**
```powershell
python train_asd_model.py
```

**Good luck! 🎯**

[![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg

# ADOS OpenPose Video Neural Network

This repository contains code for classifying Autism Spectrum Disorder (ASD) using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) normalized ADOS clinical examination video recordings. The implementation uses a VGG16-LSTM architecture for video-based classification and has been validated in peer-reviewed research ([Paper 1](https://www.nature.com/articles/s41598-021-94378-z) & [Paper 2](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308388)).

The neural network processes OpenPose skeletal keypoint videos with blank backgrounds and was originally inspired by [VideoClassifier-CNNLSTM](https://github.com/jibinmathew69/VideoClassifier-CNNLSTM).

<p align="center">
<img src=https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening/blob/main/illustrations/openpose.jpg>
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation & Environment Setup](#installation--environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Running Predictions](#running-predictions)
- [Reproducing Results](#reproducing-results)
- [Expected Outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Project Overview

### Architecture
- **Feature Extractor**: VGG16 (pre-trained on ImageNet)
- **Temporal Model**: LSTM
- **Task**: Binary classification (ASD vs TD - Typically Developing)

### Key Features
- Automated VGG16 feature extraction with caching
- LSTM-based temporal sequence modeling
- Support for video segmentation and prediction aggregation
- GPU acceleration support
- Comprehensive training history visualization

---

## Prerequisites

### Hardware Requirements
- **Minimum**: CPU-only setup (slower training)
- **Recommended**: NVIDIA GPU with CUDA support
  - CUDA Toolkit 11.2+ 
  - cuDNN 8.1+
  - 8GB+ GPU memory recommended

### Software Requirements
- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.9 - 3.13
- **Git**: For cloning the repository

### Dependencies
Core libraries (see `requirements.txt`):
- TensorFlow (with GPU support if CUDA available)
- Keras
- OpenCV
- NumPy, Pandas, Scikit-learn
- Matplotlib (for visualization)

---

## Installation & Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening.git
cd Video-Neural-Network-ASD-screening
```

### Step 2: Create Virtual Environment

**Option A: Using venv (Recommended for Windows)**

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

**Option B: Using Conda**

```bash
conda create -n asd_screening python=3.9 -y
conda activate asd_screening
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

**Check Python packages:**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

**Check GPU availability (if applicable):**
```bash
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

---

## Dataset Preparation

### Dataset Structure

The project expects datasets organized as follows:

```
your_dataset/
├── training_set/
│   ├── ASD/
│   │   ├── video1.avi
│   │   ├── video2.avi
│   │   └── ...
│   └── TD/
│       ├── video1.avi
│       ├── video2.avi
│       └── ...
└── testing_set/
    ├── ASD/
    │   └── ...
    └── TD/
        └── ...
```

### Preparing Your Dataset

1. **Video Format**: `.avi` or `.mp4` files
2. **OpenPose Processing**: Videos should be OpenPose skeletal keypoint visualizations on blank backgrounds
3. **Class Organization**: Each class (ASD, TD) in separate subdirectories
4. **Naming**: Any naming convention is supported

### Example Datasets Included

The project includes example datasets:
- `mini_dataset/` - Small sample for testing
- `dataset_20percent/` - 20% subset for development

**Update Dataset Paths:**

Edit the dataset paths in the training/prediction scripts:

**In `train_asd_model.py`:**
```python
DATASET_PATH = r"D:\projects\01\dataset\autism_data_anonymized\training_set"
```

**In `predict_asd_model.py`:**
```python
TEST_DATASET_PATH = r"D:\projects\01\dataset\autism_data_anonymized\testing_set"
```

---

## Training the Model

### Step 1: Configure Training Parameters

Open `train_asd_model.py` and adjust as needed:

```python
# Dataset location
DATASET_PATH = r"path/to/your/training_set"

# Training parameters
NUM_EPOCHS = 100          # Number of training epochs
TEST_SIZE = 0.2           # Validation split (20%)
VGG16_INCLUDE_TOP = False # Use high-dimensional features (recommended)
```

### Step 2: Run Training

```bash
python train_asd_model.py
```

### Training Process

The script will:
1. ✓ Verify GPU availability
2. ✓ Scan and validate dataset structure
3. ✓ Extract VGG16 features (or load from cache)
4. ✓ Build LSTM model
5. ✓ Train with automatic validation
6. ✓ Save best model checkpoints
7. ✓ Generate training history plots

### Monitoring Training

- **Console Output**: Real-time epoch progress, loss, and accuracy
- **Training State**: Saved to `models/autism_data/training_state.json`
- **Checkpoints**: Best models saved in `models/autism_data/checkpoints/`

### Expected Training Time

- **Small dataset (10-20 videos)**: 5-15 minutes (GPU)
- **Medium dataset (50-100 videos)**: 30-90 minutes (GPU)
- **Large dataset (100+ videos)**: 2-5 hours (GPU)

*CPU training takes significantly longer (5-10x)*

---

## Running Predictions

### Step 1: Verify Trained Model

Ensure training completed successfully:
```bash
ls models/autism_data/
# Should contain: vgg16-lstm-hi-dim-weights.h5, vgg16-lstm-hi-dim-config.npy
```

### Step 2: Prepare Test Dataset

Organize test videos in the same structure as training data:
```
testing_set/
├── ASD/
│   └── test_videos...
└── TD/
    └── test_videos...
```

### Step 3: Configure Prediction Script

Edit `predict_asd_model.py`:
```python
TEST_DATASET_PATH = r"path/to/your/testing_set"
```

### Step 4: Run Predictions

**Option A: Console output**
```bash
python predict_asd_model.py
```

**Option B: Save to CSV**
```bash
python predict_asd_model.py > predictions_output.csv
```

The script outputs predictions with confidence scores for each video.

### Understanding Predictions

Each prediction includes:
- Video filename
- Predicted class (ASD or TD)
- Confidence score (0-1)
- True label (if known)

### Prediction Aggregation

For segmented videos (5-second clips from longer recordings), you can aggregate predictions:

1. Save predictions to CSV
2. Group predictions by subject/original video
3. Apply majority voting or confidence-weighted aggregation
4. Optional: Filter by confidence threshold (e.g., >90%)

---

## Reproducing Results

To reproduce the results from the published papers:

### 1. Paper 1 (Scientific Reports 2021)

```bash
# Use the full dataset with standard 80-20 train-test split
python train_asd_model.py
python predict_asd_model.py
```

### 2. Paper 2 (PLOS ONE 2024)

This paper used video-audio ensemble models. The video component follows the same training procedure with additional confidence-based filtering during prediction aggregation (>90% threshold).

### Expected Performance Metrics

- **Accuracy**: 75-85% (varies by dataset size)
- **F1 Score**: 0.75-0.82
- **Precision**: 0.77-0.85
- **Recall**: 0.73-0.86

*Note: Results depend on dataset size, quality, and train-test split*

---

## Expected Outputs

### Training Outputs

**Directory**: `models/autism_data/`

- `vgg16-lstm-hi-dim-weights.h5` - Trained model weights
- `vgg16-lstm-hi-dim-config.npy` - Model configuration
- `vgg16-lstm-hi-dim-architecture.json` - Model architecture (JSON)
- `training_state.json` - Training metadata and history
- `checkpoints/` - Best model checkpoints by epoch

**Directory**: `reports/autism_data/`

- `vgg16-lstm-hi-dim-history.png` - Training/validation curves
- Training logs and metrics

### Prediction Outputs

- **Console**: Per-video predictions with confidence scores
- **CSV File**: `predictions_YYYYMMDD_HHMMSS.csv` with all predictions
- **Summary Statistics**: Overall accuracy, precision, recall, F1-score

### Feature Cache

**Directory**: `dataset/autism_data_anonymized/training_set-vgg16-HiDimFeatures/`

Cached VGG16 features (`.npy` files) to speed up subsequent training runs. These are automatically generated and reused.

Example training history:

<p align="center">
<img src=https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening/blob/main/reports/autism_data/vgg16-lstm-hi-dim-history.png>
</p>

---

## Troubleshooting

### Common Issues

**1. NumPy ABI version mismatch / h5py import error**
```
RuntimeError: module compiled against ABI version 0x1000009 but this version of numpy is 0x2000000
ImportError: cannot import name '_errors' from 'h5py'
```

**Solution**: NumPy 2.x is incompatible with TensorFlow/Keras. Reinstall with the correct version:
```bash
pip uninstall numpy -y
pip install "numpy<2.0"
# Or reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

**2. ModuleNotFoundError: No module named 'tensorflow'**
```bash
# Ensure virtual environment is activated
pip install -r requirements.txt
```

**3. GPU not detected despite having CUDA installed**
```bash
# Verify CUDA installation
nvidia-smi

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If still not detected, reinstall TensorFlow with correct CUDA version
pip uninstall tensorflow
pip install tensorflow
```

**4. Dataset not found errors**
- Verify the `DATASET_PATH` in your training/prediction scripts
- Check that folder structure follows the required format
- Ensure videos are in `.avi` or `.mp4` format

**5. Out of memory errors during training**
```python
# In train_asd_model.py, reduce batch size by editing:
# The batch size is auto-calculated, but you can manually set it lower
```

**6. Feature extraction taking too long**
- Features are cached automatically after first extraction
- Check `training_set-vgg16-HiDimFeatures/` for cached `.npy` files
- Using GPU significantly speeds up feature extraction

**7. Model performance is poor**
- Ensure sufficient training data (minimum 50-100 videos recommended)
- Check data balance between classes
- Increase number of epochs
- Verify video quality and OpenPose processing

**8. Permission errors on Windows**
```powershell
# Run PowerShell as administrator or adjust execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Getting Help

For additional support:
- Check the [Issues](https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening/issues) page
- Review the referenced papers for methodological details
- Ensure all prerequisites are correctly installed

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Natraj2024,
    doi = {10.1371/journal.pone.0308388},
    author = {Natraj, Shreyasvi AND Kojovic, Nada AND Maillart, Thomas AND Schaer, Marie},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Video-audio neural network ensemble for comprehensive screening of autism spectrum disorder in young children},
    year = {2024},
    month = {10},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pone.0308388},
    pages = {1-20},
    number = {10},
}

@article{Kojovic2021,
    title = {Using 2D video-based pose estimation for automated prediction of autism spectrum disorders in young children},
    author = {Kojovic, Nada and Natraj, Shreyasvi and Mohanty, Sharada Prasanna and Maillart, Thomas and Schaer, Marie},
    year = 2021,
    month = {Jul},
    day = 23,
    journal = {Scientific Reports},
    volume = 11,
    number = 1,
    pages = 15069,
    doi = {10.1038/s41598-021-94378-z},
    issn = {2045-2322},
    url = {https://doi.org/10.1038/s41598-021-94378-z}
}
```

---

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License][cc-by-nc-nd].

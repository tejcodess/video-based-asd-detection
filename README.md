# Video-Based ASD Detection Using Deep Learning

Automated detection of Autism Spectrum Disorder (ASD) from video data using VGG16 + LSTM deep learning architecture.

[![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg

## 🎯 Overview

This project uses deep learning to classify videos of children into two categories:
- **ASD**: Autism Spectrum Disorder
- **TD**: Typically Developing

The model extracts visual features using VGG16 (pre-trained on ImageNet) and uses LSTM networks to capture temporal patterns across video frames. This implementation has been validated in peer-reviewed research ([Paper 1](https://www.nature.com/articles/s41598-021-94378-z) & [Paper 2](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308388)).

<p align="center">
<img src=https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening/blob/main/illustrations/openpose.jpg>
</p>

## ✨ Features

- **VGG16 Feature Extraction**: Transfer learning from ImageNet
- **LSTM Temporal Analysis**: Captures behavioral patterns over time
- **Feature Caching**: Speeds up training by caching extracted features
- **Checkpoint System**: Save and resume training at any time
- **GPU Acceleration**: Supports NVIDIA GPUs with CUDA
- **Flexible Configuration**: Easy to customize via `config.py`
- **OpenPose Support**: Process OpenPose skeletal keypoint videos

## 📋 Requirements

- **Python**: 3.11 (recommended) or 3.9-3.11
- **GPU** (optional but recommended): NVIDIA GPU with CUDA support
- **RAM**: 16 GB minimum
- **Disk Space**: 
  - Code: < 100 MB
  - Dataset: Depends on your data (10-100 GB typical)
  - Models: ~2 GB for checkpoints

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening.git
cd Video-Neural-Network-ASD-screening
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Dataset
Organize your videos in this structure:

```
dataset/
├── training_set/
│   ├── ASD/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── TD/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
└── testing_set/
    ├── ASD/
    │   ├── video1.mp4
    │   └── ...
    └── TD/
        ├── video1.mp4
        └── ...
```

### 4. Configure Paths
Edit [config.py](config.py) and set your dataset location:

```python
DATASET_BASE = r"C:\path\to\your\dataset"
```

### 5. Train Model
```bash
python train_asd_model.py
```

The training process will:
- Extract VGG16 features from videos (cached for reuse)
- Train LSTM model on temporal patterns
- Save checkpoints every epoch
- Generate training history plots
- Save final model and configuration

### 6. Make Predictions
```bash
python predict_asd_model.py
```

Results will be saved to:
- `predictions_YYYYMMDD_HHMMSS.csv` - Detailed predictions
- Console output with accuracy metrics

## 📁 Project Structure

```
Video-Neural-Network-ASD-screening/
├── config.py                    # Configuration file (EDIT THIS FIRST!)
├── train_asd_model.py          # Training script
├── predict_asd_model.py        # Prediction script
├── asd_data_loader.py          # Data loading utilities
├── vgg16_feature_extractor.py  # Feature extraction
├── recurrent_networks.py       # LSTM model architecture
├── plot_utils.py               # Visualization utilities
├── verify_installation.py      # Check environment setup
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── INSTALLATION.md             # Detailed installation guide
├── DATASET_GUIDE.md            # Dataset preparation guide
├── LICENSE.md                  # License information
│
├── models/                     # Trained models (auto-generated)
│   └── autism_data/
│       ├── vgg16-lstm-hi-dim-weights.h5
│       ├── vgg16-lstm-hi-dim-architecture.json
│       ├── vgg16-lstm-hi-dim-config.npy
│       ├── training_state.json
│       └── checkpoints/
│
├── reports/                    # Training reports (auto-generated)
│   └── autism_data/
│       ├── training_history.png
│       └── accuracy_plot.png
│
├── logs/                       # Training logs (auto-generated)
│
└── extracted_features/         # Cached VGG16 features (auto-generated)
```

## 🔧 Configuration

All settings are in [config.py](config.py). Key parameters:

```python
# Dataset location
DATASET_BASE = r"D:\path\to\dataset"

# Model architecture
VGG16_INCLUDE_TOP = False    # False for hi-dim features (25088)
LSTM_UNITS = 512
DROPOUT_RATE = 0.5

# Training
BATCH_SIZE = 625             # Reduce if GPU out of memory
EPOCHS = 100
VALIDATION_SPLIT = 0.2       # 20% validation data
```

## 🎓 How It Works

### 1. Feature Extraction (VGG16)
- Extracts 1 frame per second from each video
- Uses VGG16 (pre-trained on ImageNet) to extract visual features
- Features are cached to disk to speed up subsequent training runs

### 2. Temporal Analysis (LSTM)
- LSTM processes the sequence of frame features
- Learns temporal patterns that distinguish ASD from TD behaviors
- Outputs binary classification: ASD or TD

### 3. Training Process
```
Video → Frame Extraction → VGG16 Features → LSTM → Classification
  |            |                  |            |          |
 .mp4      224x224 images    25088-dim      512 units  ASD/TD
```

## 📊 Expected Results

With a well-balanced dataset, you should achieve:
- **Training Accuracy**: 80-95%
- **Validation Accuracy**: 75-90%
- **Test Accuracy**: 70-85%

Results depend heavily on:
- Dataset quality and size
- Video diversity
- Class balance
- Training time

## 🐛 Troubleshooting

### GPU Out of Memory
```python
# In config.py, reduce batch size
BATCH_SIZE = 256  # or 128, or 64
```

### CUDA/cuDNN Not Found
```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]==2.15.0
```

### NumPy ABI version mismatch
```bash
# NumPy 2.x is incompatible with TensorFlow 2.15
pip uninstall numpy -y
pip install "numpy<2.0"
```

### Import Errors
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### Feature Extraction Slow
Feature extraction runs once and caches results. Subsequent runs use cached features.
- First run: ~1-2 hours (depending on dataset size)
- Subsequent runs: ~5-10 minutes

### Model Not Learning
Check:
1. Dataset is properly organized (ASD/ and TD/ folders)
2. Sufficient training data (>100 videos per class minimum)
3. Videos are not corrupted
4. Classes are balanced

## 🔬 Advanced Usage

### Resume Training from Checkpoint
Training automatically saves checkpoints. If interrupted, just run:
```bash
python train_asd_model.py
```
The script will detect existing checkpoints and ask if you want to resume.

### Custom Dataset Structure
If your dataset has a different structure, modify paths in [config.py](config.py):
```python
TRAINING_DATA_PATH = r"C:\custom\path\train"
TESTING_DATA_PATH = r"C:\custom\path\test"
CLASSES = ["ASD", "TD"]  # Must match folder names
```

### Hyperparameter Tuning
Edit [config.py](config.py) to experiment:
- `LSTM_UNITS`: 256, 512, 1024
- `DROPOUT_RATE`: 0.3, 0.5, 0.7
- `BATCH_SIZE`: 64, 128, 256, 625
- `LEARNING_RATE`: Add custom learning rate in training script

### Feature Dimension Selection
```python
# Standard VGG16 features (1000-dim)
VGG16_INCLUDE_TOP = True

# Hi-dimensional features (25088-dim) - RECOMMENDED
VGG16_INCLUDE_TOP = False
```

## 📖 Documentation

- [INSTALLATION.md](INSTALLATION.md) - Complete installation guide
- [DATASET_GUIDE.md](DATASET_GUIDE.md) - How to prepare your dataset
- [LICENSE.md](LICENSE.md) - License information

## 🧪 Verify Installation

Before training, verify your environment:
```bash
python verify_installation.py
```

This checks:
- Python version
- TensorFlow installation
- GPU availability
- Required packages
- Dataset paths

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## ⚠️ Privacy & Ethics

This tool is for research purposes only. When working with sensitive medical data:

1. **Obtain proper consent** from participants or guardians
2. **Anonymize data** - remove personally identifiable information
3. **Secure storage** - encrypt sensitive data
4. **Follow regulations** - HIPAA, GDPR, local laws
5. **Clinical validation** - not a replacement for professional diagnosis

## 📄 License

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License][cc-by-nc-nd].

## 🙏 Acknowledgments

- **VGG16**: Visual Geometry Group, University of Oxford
- **ImageNet**: Pre-trained weights for transfer learning
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision library
- **Inspired by**: [VideoClassifier-CNNLSTM](https://github.com/jibinmathew69/VideoClassifier-CNNLSTM)

## 📬 Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening/issues](https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening/issues)

## 📚 Citation

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

## 📚 References

1. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

3. Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. CVPR 2009.

---

**Made with ❤️ for autism research and early intervention**

*Last updated: February 2026*

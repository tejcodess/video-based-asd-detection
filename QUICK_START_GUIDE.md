# Quick Start Guide - ASD Screening Project

Get a working prototype running in **30 minutes** with ability to scale to full training.

---

## 🎯 Overview

This project trains deep learning models (VGG16 + LSTM) to classify videos for ASD screening:
- **Mini Test**: 10 videos, 5-15 minutes, ~500 MB
- **Full Training**: 9,680 videos, 8-15 hours, ~100 GB

---

## ✅ Prerequisites

### Required
- **Windows 10/11**
- **Python 3.11** (NOT 3.13) - [Download here](https://www.python.org/downloads/release/python-3118/)
- **Disk Space**: 2 GB for mini test, 150 GB for full training
- **Dataset**: Already at `D:\projects\01\dataset\autism_data_anonymized\`

### Optional (Recommended)
- **NVIDIA GPU** with 6+ GB VRAM (RTX 3050 or better)
- **CUDA 12.x** drivers installed

---

## 🚀 Step 1: Install Python 3.11

If you have Python 3.13, you need Python 3.11 instead.

### Download
Visit: https://www.python.org/downloads/release/python-3118/

Click: **Windows installer (64-bit)**

### Install
1. Run the installer
2. ✅ **CHECK "Add Python 3.11 to PATH"** (important!)
3. Click "Install Now"
4. Restart PowerShell after installation

### Verify
```powershell
python --version
```
Should show: `Python 3.11.8`

---

## 🔧 Step 2: Set Up Environment

Open PowerShell in the project folder and run:

```powershell
.\setup_venv.ps1
```

**What it does:**
- ✓ Checks Python 3.11 is installed
- ✓ Creates virtual environment `venv_asd`
- ✓ Installs TensorFlow 2.15
- ✓ Installs all dependencies
- ✓ Tests GPU detection

**Expected time:** 5-10 minutes  
**Downloads:** ~500 MB

### Expected Output
```
[1/9] Checking Python version...
✓ Python 3.11 detected - PERFECT!

[2/9] Creating virtual environment 'venv_asd'...
✓ Virtual environment created

[5/9] Installing TensorFlow 2.15.0...
✓ TensorFlow installed

[8/9] Testing GPU detection...
  GPUs detected: 1
  - /physical_device:GPU:0
```

---

## 📦 Step 3: Create Mini Test Dataset

Activate the environment and create a tiny dataset for testing:

```powershell
.\venv_asd\Scripts\Activate.ps1
python create_mini_dataset.py
```

**What it does:**
- Randomly selects 5 ASD + 5 TD videos for training
- Randomly selects 3 ASD + 3 TD videos for testing
- Copies to `mini_dataset/` folder

**Expected time:** 1 minute  
**Disk space:** ~500 MB

### Expected Output
```
CREATING MINI DATASET FOR QUICK TESTING
========================================
Training set: 5 ASD + 5 TD = 10 videos
Testing set:  3 ASD + 3 TD = 6 videos
Total: 16 videos

Location: D:\projects\01\Video-Neural-Network-ASD-screening\mini_dataset
```

---

## 🧪 Step 4: Run Quick Test

Train a model on the mini dataset:

```powershell
python quick_test.py
```

**What it does:**
1. Loads mini dataset (10 videos)
2. Extracts VGG16 features from each video
3. Trains VGG16-LSTM model (100 epochs)
4. Saves model checkpoint
5. Generates accuracy/loss plot

**Expected time:**
- With GPU: 5-15 minutes
- With CPU: 30-60 minutes

### Expected Output
```
QUICK TEST - VGG16-LSTM TRAINING
================================
✓ Using mini dataset (fast test mode)

[STEP 1/5] Loading dataset...
✓ Dataset loaded

[STEP 2/5] Creating VGG16-LSTM model...
✓ Model created

[STEP 3/5] Training model...
Epoch 1/100
loss: 0.693 - accuracy: 0.500
...
Epoch 100/100
loss: 0.234 - accuracy: 0.900

[STEP 5/5] Test complete!
Total time: 12.3 minutes
```

---

## 🎉 Success!

You now have a trained model! Check the files:

```
models/autism_data/
├── vgg16-lstm-hi-dim-config.npy
├── vgg16-lstm-hi-dim-weights.h5
└── vgg16-lstm-hi-dim-architecture.json

reports/autism_data/
└── quick-test-history.png  ← Training accuracy plot
```

---

## 🚀 Step 5: Switch to Full Training (Optional)

Once the quick test works, scale to the full dataset:

### 5.1 Enable Full Dataset

Edit [config.py](config.py), line 8:

```python
USE_MINI_DATASET = False  # Changed from True
```

### 5.2 Run Full Training

```powershell
python train_simple.py
```

**Expected:**
- **Time:** 8-15 hours on GPU
- **Disk space:** ~100 GB for cached features
- **Model size:** ~500 MB

### 5.3 Monitor Progress

In a separate PowerShell window:

```powershell
# Watch GPU usage
nvidia-smi -l 1

# Watch training logs
Get-Content reports/autism_data/training.log -Wait
```

### 5.4 Training Output

The model saves checkpoints every epoch. If interrupted, training resumes from the last checkpoint.

---

## 🐛 Troubleshooting

### "Python was not found"

**Problem:** Python not in PATH

**Solution:**
1. Reinstall Python 3.11
2. ✅ Check "Add Python 3.11 to PATH"
3. Restart PowerShell

---

### "TensorFlow not found"

**Problem:** Virtual environment not activated or packages not installed

**Solution:**
```powershell
.\venv_asd\Scripts\Activate.ps1
.\setup_venv.ps1
```

---

### "No GPU detected"

**Problem:** GPU drivers not installed or incompatible

**Check GPU:**
```powershell
nvidia-smi
```

If this fails, install NVIDIA drivers from:
https://www.nvidia.com/Download/index.aspx

**Note:** Training works on CPU (just slower)

---

### "Out of memory"

**Problem:** GPU VRAM insufficient (< 6 GB)

**Solution 1:** Reduce batch size

Edit [recurrent_networks.py](recurrent_networks.py), line ~200:

```python
BATCH_SIZE = 256  # Changed from 625
```

**Solution 2:** Use CPU

GPU memory growth is enabled by default, but for very large models:

Edit [gpu_utils.py](gpu_utils.py):

```python
configure_gpu(memory_growth=True, memory_limit=4096)  # 4 GB limit
```

---

### "Dataset not found"

**Problem:** Dataset path incorrect

**Solution:** Edit [config.py](config.py):

```python
DATASET_BASE = r"YOUR_ACTUAL_PATH_HERE"
```

---

### Training is very slow

**Check GPU usage:**
```powershell
nvidia-smi -l 1
```

**Expected:** GPU Utilization: 80-100%

**If low:** 
- Check if TensorFlow detects GPU:
  ```powershell
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  ```
- May need to reinstall TensorFlow with GPU support

---

## 📁 Project Structure

```
Video-Neural-Network-ASD-screening/
├── config.py                    ← Dataset paths & settings
├── gpu_utils.py                 ← GPU configuration
├── recurrent_networks.py        ← Model definitions
├── vgg16_feature_extractor.py   ← Feature extraction
├── UCF101_loader.py             ← Dataset loading
├── plot_utils.py                ← Plotting utilities
│
├── setup_venv.ps1               ← Environment setup script
├── create_mini_dataset.py       ← Mini dataset creator
├── quick_test.py                ← Quick end-to-end test
│
├── train_simple.py              ← Full training script
├── predict_simple.py            ← Prediction script
│
├── models/autism_data/          ← Saved models
├── reports/autism_data/         ← Training plots
├── extracted_features/          ← Cached VGG16 features
└── mini_dataset/                ← Mini test dataset
```

---

## 🔄 Typical Workflow

### First Time (Quick Test)
```powershell
# 1. Setup (once)
.\setup_venv.ps1

# 2. Create mini dataset
python create_mini_dataset.py

# 3. Quick test
python quick_test.py
```

### Full Training
```powershell
# 1. Edit config.py: USE_MINI_DATASET = False

# 2. Train on full dataset
python train_simple.py

# 3. Make predictions
python predict_simple.py
```

### Resume Training
If training is interrupted, just run again:
```powershell
python train_simple.py
```
Cached features are reused, so it's much faster.

---

## 📊 Expected Results

### Mini Dataset (Quick Test)
- **Training Accuracy:** 80-100% (small dataset, may overfit)
- **Time:** 5-15 minutes
- **Purpose:** Verify everything works

### Full Dataset
- **Training Accuracy:** 85-95%
- **Test Accuracy:** 75-90% (depends on data quality)
- **Time:** 8-15 hours
- **Purpose:** Production model

---

## 🔗 Additional Resources

- **Full Documentation:** [PROJECT_COMPLETE_ANALYSIS.txt](PROJECT_COMPLETE_ANALYSIS.txt)
- **GPU Setup:** [GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md)
- **Project Updates:** [PROJECT_UPDATE_GUIDE.md](PROJECT_UPDATE_GUIDE.md)

---

## 💡 Tips

1. **Always use mini dataset first** to verify everything works
2. **Monitor GPU usage** during training (`nvidia-smi -l 1`)
3. **Cached features** make subsequent runs much faster
4. **Save checkpoints** - training can be resumed if interrupted
5. **Check plots** in `reports/autism_data/` to monitor training

---

## 🆘 Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review troubleshooting section above
3. Check [PROJECT_COMPLETE_ANALYSIS.txt](PROJECT_COMPLETE_ANALYSIS.txt) for details
4. Verify dataset structure matches expected format

---

## ✨ Next Steps After Training

1. **Evaluate Model:**
   ```powershell
   python predict_simple.py
   ```

2. **Check Results:**
   - Confusion matrix
   - Per-class accuracy
   - Misclassified videos

3. **Fine-tune:**
   - Adjust learning rate in `recurrent_networks.py`
   - Try different architectures
   - Add data augmentation

4. **Deploy:**
   - Export model for production
   - Create inference API
   - Integrate with application

---

**Good luck with your ASD screening project!** 🚀

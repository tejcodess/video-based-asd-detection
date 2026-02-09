# Installation Guide

Complete step-by-step guide to install and configure the ASD Detection Model on Windows, Linux, and macOS.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Install Python](#step-1-install-python)
- [Step 2: Clone Repository](#step-2-clone-repository)
- [Step 3: Create Virtual Environment](#step-3-create-virtual-environment)
- [Step 4: Install Dependencies](#step-4-install-dependencies)
- [Step 5: GPU Setup (Optional)](#step-5-gpu-setup-optional)
- [Step 6: Verify Installation](#step-6-verify-installation)
- [Step 7: Configure Dataset Paths](#step-7-configure-dataset-paths)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements

- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **RAM**: 16 GB minimum (32 GB recommended)
- **Disk Space**: 20 GB free space
- **Internet**: Required for downloading dependencies

### Optional (for GPU acceleration)

- **NVIDIA GPU**: GTX 1060 or better (8GB+ VRAM recommended)
- **CUDA**: Version 11.2 or 12.x
- **cuDNN**: Version 8.1+

---

## Step 1: Install Python

### Windows

**Option A: Download from python.org (Recommended)**

1. Download Python 3.11 from [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Run the installer
3. **IMPORTANT**: Check "Add Python to PATH"
4. Click "Install Now"
5. Verify installation:

```powershell
python --version
# Should show: Python 3.11.x
```

**Option B: Using Microsoft Store**

1. Open Microsoft Store
2. Search for "Python 3.11"
3. Click "Get" to install

**Option C: Using Chocolatey**

```powershell
choco install python311 -y
```

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Verify installation
python3.11 --version
```

### macOS

**Option A: Using Homebrew (Recommended)**

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Verify installation
python3.11 --version
```

**Option B: Download from python.org**

1. Download from [https://www.python.org/downloads/macos/](https://www.python.org/downloads/macos/)
2. Run the .pkg installer
3. Follow installation prompts

---

## Step 2: Clone Repository

### Using Git (Recommended)

```bash
# Install Git if not installed
# Windows: https://git-scm.com/download/win
# Linux: sudo apt install git
# macOS: brew install git

# Clone repository
git clone https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening.git

# Navigate to directory
cd Video-Neural-Network-ASD-screening
```

### Manual Download

1. Go to [https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening](https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening)
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file
4. Open terminal/command prompt in the extracted folder

---

## Step 3: Create Virtual Environment

Creating a virtual environment isolates project dependencies from your system Python.

### Windows

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then activate again
```

### Linux/macOS

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Verify Activation

After activation, your prompt should show `(venv)`:

```
(venv) PS D:\projects\Video-Neural-Network-ASD-screening>
```

---

## Step 4: Install Dependencies

### Install Required Packages

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### Installation Progress

This will install:
- TensorFlow 2.15.0 (~500 MB)
- OpenCV (~50 MB)
- NumPy, Pandas, Scikit-learn
- Matplotlib, imutils, tqdm

**Expected time**: 5-15 minutes (depending on internet speed)

### Verify Installation

```bash
# Check TensorFlow
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"

# Check OpenCV
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# Check NumPy
python -c "import numpy as np; print('NumPy:', np.__version__)"
```

Expected output:
```
TensorFlow: 2.15.0
OpenCV: 4.8.0.74
NumPy: 1.24.3
```

---

## Step 5: GPU Setup (Optional)

GPU acceleration speeds up training by 5-10x. Skip this section if you don't have an NVIDIA GPU.

### Check GPU Availability

```bash
# Check if NVIDIA GPU is detected
nvidia-smi
```

If this command works, you have an NVIDIA GPU and driver installed.

### Install CUDA and cuDNN

#### Windows

**Option 1: Automatic (Recommended for TensorFlow 2.15)**

TensorFlow 2.15 can automatically install CUDA/cuDNN:

```bash
pip install tensorflow[and-cuda]==2.15.0
```

**Option 2: Manual Installation**

1. Download CUDA Toolkit 12.x from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Download cuDNN 8.x from [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn) (requires free NVIDIA account)
3. Install CUDA Toolkit
4. Extract cuDNN and copy files to CUDA directory:
   - Copy `cudnn*/bin/*` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
   - Copy `cudnn*/include/*` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include`
   - Copy `cudnn*/lib/*` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib`

#### Linux

```bash
# Install CUDA (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt install cuda -y

# Install cuDNN
sudo apt install libcudnn8 libcudnn8-dev -y
```

### Verify GPU Setup

```bash
# Check if TensorFlow detects GPU
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"
```

Expected output:
```
GPU Available: True
GPU Devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Configure GPU Memory Growth

GPU memory growth is enabled by default in [config.py](config.py):

```python
GPU_MEMORY_GROWTH = True  # Prevents OOM errors
GPU_MEMORY_LIMIT = None   # Set to 4096 for 4GB limit
```

---

## Step 6: Verify Installation

Use the built-in verification script:

```bash
python verify_installation.py
```

This checks:
- ✓ Python version (3.9-3.11)
- ✓ TensorFlow installation
- ✓ GPU availability
- ✓ Required packages
- ✓ Directory structure

Expected output:
```
============================================================
ENVIRONMENT VERIFICATION
============================================================
Python version:      3.11.x ✓
TensorFlow:          2.15.0 ✓
Keras:               2.15.0 ✓
OpenCV:              4.8.0.74 ✓
NumPy:               1.24.3 ✓
GPU Available:       True ✓
CUDA Version:        12.x ✓
============================================================
✓ All checks passed! Ready to train models.
============================================================
```

---

## Step 7: Configure Dataset Paths

Edit [config.py](config.py) to point to your dataset:

```python
# Set this to your dataset location
DATASET_BASE = r"D:\projects\01\dataset\autism_data_anonymized"

# These are automatically set based on DATASET_BASE
TRAINING_DATA_PATH = os.path.join(DATASET_BASE, "training_set")
TESTING_DATA_PATH = os.path.join(DATASET_BASE, "testing_set")
```

### Verify Configuration

```bash
# Print current configuration
python config.py
```

This will show:
- Dataset paths
- Model configuration
- Training parameters

---

## Troubleshooting

### NumPy Version Mismatch

**Error**: `RuntimeError: module compiled against ABI version 0x1000009 but this version of numpy is 0x2000000`

**Solution**:
```bash
pip uninstall numpy -y
pip install "numpy<2.0"
```

### TensorFlow Import Error

**Error**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solutions**:

1. Ensure virtual environment is activated (you should see `(venv)` in prompt)
2. Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

### GPU Not Detected

**Error**: GPU available but TensorFlow doesn't detect it

**Solutions**:

1. Check NVIDIA driver:
```bash
nvidia-smi
```

2. Reinstall TensorFlow with GPU support:
```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.15.0
```

3. Verify CUDA/cuDNN versions are compatible with TensorFlow 2.15:
   - CUDA 11.2+ or 12.x
   - cuDNN 8.1+

### Permission Errors (Windows)

**Error**: `cannot be loaded because running scripts is disabled`

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### OpenCV Import Error

**Error**: `ImportError: DLL load failed while importing cv2`

**Solution** (Windows):
```bash
pip uninstall opencv-contrib-python opencv-python -y
pip install opencv-contrib-python==4.8.0.74
```

### Disk Space Issues

**Error**: Not enough disk space during installation

**Solution**:
- Free up at least 20 GB
- Install dependencies one at a time:
```bash
pip install tensorflow==2.15.0
pip install opencv-contrib-python==4.8.0.74
pip install numpy==1.24.3
# ... continue with others
```

### Slow Installation

If pip installation is slow:

```bash
# Use faster mirror (China users)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Or use local cache
pip install --cache-dir ./pip_cache -r requirements.txt
```

### Virtual Environment Not Activating

**Windows**: Use Command Prompt instead of PowerShell:
```cmd
venv\Scripts\activate.bat
```

**Linux/macOS**: Ensure you're using bash/zsh:
```bash
source venv/bin/activate
```

---

## Next Steps

After successful installation:

1. **Prepare dataset**: See [DATASET_GUIDE.md](DATASET_GUIDE.md)
2. **Configure paths**: Edit [config.py](config.py)
3. **Train model**: Run `python train_asd_model.py`
4. **Make predictions**: Run `python predict_asd_model.py`

---

## Platform-Specific Notes

### Windows

- Use PowerShell or Command Prompt (not Git Bash)
- Paths use backslashes `\` or raw strings `r"path\to\file"`
- If using Anaconda, prefer `conda` over `venv`

### Linux

- May need `sudo` for system-wide package installation
- Use `python3.11` explicitly if multiple Python versions installed
- For GPU: Ensure NVIDIA drivers are up to date

### macOS

- Apple Silicon (M1/M2) users: Use tensorflow-metal for acceleration
```bash
pip install tensorflow-metal
```
- Intel Macs: Follow standard GPU setup (if NVIDIA eGPU)

---

## Additional Resources

- **TensorFlow Installation**: [https://www.tensorflow.org/install](https://www.tensorflow.org/install)
- **CUDA Installation**: [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
- **Python Virtual Environments**: [https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html)

---

## Getting Help

If you encounter issues not covered here:

1. Check [README.md](README.md) troubleshooting section
2. Search [GitHub Issues](https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening/issues)
3. Create a new issue with:
   - Your OS and Python version
   - Error message (full stack trace)
   - Output of `pip list`
   - Steps to reproduce

---

**Installation complete!** You're now ready to train ASD detection models.

*For dataset preparation, see [DATASET_GUIDE.md](DATASET_GUIDE.md)*

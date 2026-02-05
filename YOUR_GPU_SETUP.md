# GPU Setup for Your System

## Your Hardware
- **GPU**: NVIDIA GeForce RTX 3050 Laptop (6GB VRAM)
- **Current CUDA**: Version 12.9 (shown in nvidia-smi)
- **Driver**: 576.80

## Issue
TensorFlow 2.10.0 requires **CUDA 11.2**, but your system has CUDA 12.9.

## Solution Options

### Option 1: Upgrade to TensorFlow 2.15+ (Recommended)
TensorFlow 2.15+ supports CUDA 12.x, which you already have!

**Steps:**
```powershell
# Upgrade TensorFlow and Keras
pip install --upgrade tensorflow==2.15.0
pip install --upgrade keras==2.15.0

# Test GPU
python gpu_utils.py
```

**Pros:**
- Uses your existing CUDA 12.9 installation
- No additional software needed
- Newer TensorFlow version with bug fixes
- Should work immediately

**Cons:**
- Slightly different API (but our code should work)

### Option 2: Install CUDA 11.2 alongside CUDA 12.9
You can have multiple CUDA versions installed.

**Steps:**
1. Download CUDA 11.2 from: https://developer.nvidia.com/cuda-11.2.0-download-archive
2. Install it to a different directory (e.g., C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2)
3. Download cuDNN 8.1 for CUDA 11.2 from: https://developer.nvidia.com/cudnn
4. Extract cuDNN files to CUDA 11.2 directory
5. Set environment variables to point to CUDA 11.2:
   ```powershell
   $env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
   $env:PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin;" + $env:PATH
   ```

**Pros:**
- Keep both CUDA versions
- Works with TensorFlow 2.10.0 as specified

**Cons:**
- More complex setup
- Need to manage environment variables
- Potential conflicts

### Option 3: Use CPU for Now (Slower but Works)
The current setup will work with CPU - it'll just be slower.

**No changes needed** - training will automatically use CPU.

**Performance:**
- CPU: ~2-4 hours for 100 videos
- GPU: ~20-40 minutes for 100 videos

## Recommended: Option 1 - Upgrade TensorFlow

This is the easiest and best option for your system.

### Installation Commands
```powershell
# Uninstall old versions
pip uninstall tensorflow keras -y

# Install TensorFlow 2.15 with CUDA 12 support
pip install tensorflow==2.15.0
pip install keras==2.15.0

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')"

# Test GPU
python gpu_utils.py
```

### Expected Output
```
================================================================================
GPU INFORMATION
================================================================================

Found 1 GPU(s):

GPU 0: /physical_device:GPU:0
  device_name: NVIDIA GeForce RTX 3050 Laptop GPU

TensorFlow GPU Support:
  Built with CUDA: True
  GPU Available: True

Logical GPUs: 1
================================================================================
```

## After GPU Setup

Once GPU is working, run:
```powershell
# Verify everything
python verify_setup.py

# Train with GPU
python train_simple.py
```

You should see:
```
================================================================================
GPU CONFIGURATION
================================================================================
✓ Enabled memory growth for /physical_device:GPU:0
✓ GPU Configuration Successful!
  Device: /physical_device:GPU:0
  Memory Growth: True
================================================================================

✓ Training will use GPU acceleration
```

## During Training

Open another terminal and monitor GPU usage:
```powershell
nvidia-smi -l 1
```

You should see:
- GPU Utilization: 70-90%
- Memory Usage: 2-5GB (out of 6GB available)
- Temperature: 60-80°C (normal)

## If GPU Still Not Detected After TF 2.15 Upgrade

Try these troubleshooting steps:

1. **Check Python architecture** (must be 64-bit):
   ```powershell
   python -c "import struct; print(struct.calcsize('P') * 8, 'bit')"
   ```
   Should output: `64 bit`

2. **Verify CUDA is in PATH**:
   ```powershell
   nvcc --version
   ```

3. **Check cuDNN**:
   ```powershell
   python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
   ```

4. **Force GPU visibility**:
   ```powershell
   $env:CUDA_VISIBLE_DEVICES="0"
   python gpu_utils.py
   ```

5. **Clean reinstall**:
   ```powershell
   pip uninstall tensorflow keras -y
   pip cache purge
   pip install tensorflow==2.15.0 keras==2.15.0
   ```

## Your RTX 3050 Laptop GPU Performance

With 6GB VRAM, you can:
- ✅ Train this ASD model (uses ~2-4GB during training)
- ✅ Use default batch size (625)
- ✅ Process videos up to 1080p
- ⚠️ May need to reduce batch size for very large datasets

Expected training time: **20-40 minutes** for 100 videos (vs 2-4 hours on CPU)

## Summary

**Action Required:**
```powershell
pip uninstall tensorflow keras -y
pip install tensorflow==2.15.0 keras==2.15.0
python gpu_utils.py
```

This should enable GPU support immediately with your existing CUDA 12.9 installation!

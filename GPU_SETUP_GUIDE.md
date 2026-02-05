# GPU Setup and Usage Guide

## Quick GPU Check

Run this to check if your GPU is detected:
```powershell
python gpu_utils.py
```

Or use the full setup verification:
```powershell
python verify_setup.py
```

## GPU Requirements

For TensorFlow to use your GPU, you need:

1. **NVIDIA GPU** with CUDA Compute Capability 3.5 or higher
   - Check your GPU: https://developer.nvidia.com/cuda-gpus

2. **CUDA Toolkit 11.2+**
   - Download: https://developer.nvidia.com/cuda-toolkit-archive
   - Recommended: CUDA 11.2

3. **cuDNN 8.1+**
   - Download: https://developer.nvidia.com/cudnn
   - Requires NVIDIA Developer account (free)

4. **Updated NVIDIA Drivers**
   - Download: https://www.nvidia.com/download/index.aspx

## Installation Steps

### Step 1: Install NVIDIA Drivers
```powershell
# Check current driver version
nvidia-smi
```

### Step 2: Install CUDA Toolkit
1. Download CUDA Toolkit 11.2 from NVIDIA
2. Run installer
3. Add to PATH (installer usually does this automatically)

### Step 3: Install cuDNN
1. Download cuDNN 8.1 for CUDA 11.2
2. Extract files
3. Copy files to CUDA directory:
   ```
   cudnn/bin/*.dll     → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
   cudnn/include/*.h   → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include
   cudnn/lib/*.lib     → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib
   ```

### Step 4: Install TensorFlow
```powershell
pip install tensorflow==2.10.0
```

### Step 5: Verify GPU
```powershell
python gpu_utils.py
```

## Training with GPU

The training scripts automatically detect and use GPU if available:

```powershell
# Training will use GPU automatically
python train_simple.py
```

You'll see output like:
```
================================================================================
GPU INFORMATION
================================================================================
Found 1 GPU(s):
GPU 0: /physical_device:GPU:0
...
================================================================================
GPU CONFIGURATION
================================================================================
✓ Enabled memory growth for /physical_device:GPU:0
✓ GPU Configuration Successful!
```

## GPU Configuration Options

The training scripts use these GPU settings:

1. **Memory Growth**: Enabled by default
   - TensorFlow allocates GPU memory as needed
   - Prevents allocating all GPU memory at once
   - Allows multiple processes to use GPU

2. **Memory Limit**: Not set by default
   - Uses all available GPU memory
   - Can be limited if needed (see gpu_utils.py)

## Troubleshooting

### "No GPU detected"
**Possible causes:**
- CUDA not installed
- cuDNN not installed
- Wrong CUDA version (need 11.2+)
- NVIDIA drivers outdated
- GPU not CUDA-compatible

**Solution:**
1. Run `nvidia-smi` to verify GPU is recognized
2. Check CUDA installation: `nvcc --version`
3. Reinstall CUDA/cuDNN if needed

### "Could not load dynamic library 'cudart64_110.dll'"
**Solution:**
- Install CUDA Toolkit 11.2
- Add CUDA bin directory to PATH

### "Could not load dynamic library 'cudnn64_8.dll'"
**Solution:**
- Install cuDNN 8.1
- Copy DLL files to CUDA bin directory

### Training is slow despite GPU detected
**Possible causes:**
- GPU not actually being used
- Batch size too small
- I/O bottleneck (loading data from disk)

**Solution:**
1. Check GPU usage: `nvidia-smi` in another terminal
2. Increase batch size if memory allows
3. Ensure features are cached (run training twice)

## Performance Comparison

Typical training times (100 videos, 100 epochs):

| Hardware | Time |
|----------|------|
| CPU (Intel i7) | 2-4 hours |
| GPU (GTX 1060) | 20-40 minutes |
| GPU (RTX 3070) | 10-20 minutes |
| GPU (RTX 4090) | 5-10 minutes |

GPU provides approximately **5-10x speedup** for this type of model.

## Monitoring GPU Usage

### During Training
Open a new terminal and run:
```powershell
nvidia-smi -l 1
```

This shows:
- GPU utilization %
- Memory usage
- Temperature
- Power consumption

### Expected GPU Usage
- **Feature Extraction**: 80-100% GPU usage
- **Training**: 70-90% GPU usage
- **Prediction**: 60-80% GPU usage

## Multi-GPU Setup

If you have multiple GPUs, the code uses GPU 0 by default.

To use a specific GPU:
```python
# In gpu_utils.py, modify configure_gpu() call:
configure_gpu(memory_growth=True, gpu_id=1)  # Use GPU 1
```

Or set environment variable:
```powershell
$env:CUDA_VISIBLE_DEVICES="1"  # Use GPU 1
python train_simple.py
```

## Disabling GPU (Force CPU)

To force CPU usage (for debugging):
```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
python train_simple.py
```

## Common GPU Models Performance

| GPU Model | VRAM | Typical Performance |
|-----------|------|---------------------|
| GTX 1050 Ti | 4GB | Good for small datasets |
| GTX 1060 | 6GB | Good for medium datasets |
| RTX 2060 | 6GB | Very good |
| RTX 3060 | 12GB | Excellent |
| RTX 3070 | 8GB | Excellent |
| RTX 3080 | 10GB | Outstanding |
| RTX 4090 | 24GB | Maximum performance |

**Recommendation:** Minimum 6GB VRAM for this project.

## Memory Management

If you get "Out of Memory" errors:

1. **Reduce batch size** (in recurrent_networks.py):
   ```python
   BATCH_SIZE = 256  # Try 128 or 64
   ```

2. **Enable memory growth** (already enabled):
   ```python
   configure_gpu(memory_growth=True)
   ```

3. **Set memory limit**:
   ```python
   configure_gpu(memory_growth=True, memory_limit=6000)  # 6GB limit
   ```

4. **Clear GPU memory**:
   ```python
   import gc
   from keras import backend as K
   K.clear_session()
   gc.collect()
   ```

## Summary

✅ GPU support is **fully integrated** and enabled by default
✅ Training scripts **automatically detect and use GPU**
✅ Memory growth **enabled** to prevent OOM errors
✅ All you need is: **CUDA 11.2+ and cuDNN 8.1+**

For verification: `python gpu_utils.py`

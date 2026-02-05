"""
GPU Utilities for TensorFlow/Keras
Handles GPU detection, configuration, and memory management
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import tensorflow as tf
from keras import backend as K


def configure_gpu(memory_growth=True, memory_limit=None, gpu_id=0):
    """
    Configure GPU settings for optimal performance.
    
    Args:
        memory_growth (bool): If True, allocate GPU memory as needed (recommended)
        memory_limit (int): Maximum GPU memory in MB (None = no limit)
        gpu_id (int): Which GPU to use (0 for first GPU)
    
    Returns:
        bool: True if GPU configured successfully, False otherwise
    """
    print("\n" + "="*80)
    print("GPU CONFIGURATION")
    print("="*80)
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("⚠ No GPU detected. Training will use CPU (slower).")
        print("  For GPU support, ensure:")
        print("  - NVIDIA GPU with CUDA support installed")
        print("  - CUDA Toolkit 11.2+ installed")
        print("  - cuDNN 8.1+ installed")
        print("="*80 + "\n")
        return False
    
    try:
        # Select specific GPU if multiple available
        if len(gpus) > 1:
            print(f"Found {len(gpus)} GPUs. Using GPU {gpu_id}")
            gpus_to_use = [gpus[gpu_id]]
        else:
            gpus_to_use = gpus
        
        for gpu in gpus_to_use:
            # Enable memory growth (prevents TensorFlow from allocating all GPU memory)
            if memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Enabled memory growth for {gpu.name}")
            
            # Set memory limit if specified
            if memory_limit:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                print(f"✓ Set memory limit to {memory_limit} MB for {gpu.name}")
        
        # Set visible devices
        tf.config.set_visible_devices(gpus_to_use, 'GPU')
        
        # Print GPU information
        print(f"\n✓ GPU Configuration Successful!")
        print(f"  Device: {gpus_to_use[0].name}")
        print(f"  Memory Growth: {memory_growth}")
        if memory_limit:
            print(f"  Memory Limit: {memory_limit} MB")
        
        # Try to get GPU details
        try:
            gpu_details = tf.config.experimental.get_device_details(gpus_to_use[0])
            if 'device_name' in gpu_details:
                print(f"  GPU Name: {gpu_details['device_name']}")
        except:
            pass
        
        print("="*80 + "\n")
        return True
        
    except RuntimeError as e:
        print(f"✗ Error configuring GPU: {e}")
        print("  Training will use CPU instead.")
        print("="*80 + "\n")
        return False


def print_gpu_info():
    """Print detailed GPU information"""
    print("\n" + "="*80)
    print("GPU INFORMATION")
    print("="*80)
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPUs detected.")
        print("\nSystem will use CPU for training.")
        print("="*80 + "\n")
        return
    
    print(f"\nFound {len(gpus)} GPU(s):\n")
    
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu.name}")
        
        try:
            # Get device details
            details = tf.config.experimental.get_device_details(gpu)
            for key, value in details.items():
                print(f"  {key}: {value}")
        except:
            print("  (Details not available)")
        
        print()
    
    # Check if GPU is being used
    print("TensorFlow GPU Support:")
    print(f"  Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"  GPU Available: {tf.test.is_gpu_available()}")
    
    # Memory info
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"\nLogical GPUs: {len(logical_gpus)}")
    
    print("="*80 + "\n")


def test_gpu():
    """Test GPU by running a simple computation"""
    print("\n" + "="*80)
    print("GPU PERFORMANCE TEST")
    print("="*80)
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPU available for testing.")
        print("="*80 + "\n")
        return False
    
    try:
        import time
        
        # Test computation on GPU
        print("\nRunning matrix multiplication test...")
        with tf.device('/GPU:0'):
            # Create large matrices
            matrix_size = 5000
            a = tf.random.normal([matrix_size, matrix_size])
            b = tf.random.normal([matrix_size, matrix_size])
            
            # Warm up
            _ = tf.matmul(a, b)
            
            # Time GPU computation
            start_time = time.time()
            c = tf.matmul(a, b)
            _ = c.numpy()  # Force execution
            gpu_time = time.time() - start_time
        
        print(f"✓ GPU computation successful!")
        print(f"  Matrix size: {matrix_size}x{matrix_size}")
        print(f"  Time: {gpu_time:.4f} seconds")
        
        # Compare with CPU if available
        try:
            with tf.device('/CPU:0'):
                start_time = time.time()
                c_cpu = tf.matmul(a, b)
                _ = c_cpu.numpy()
                cpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time
            print(f"\n  CPU time: {cpu_time:.4f} seconds")
            print(f"  GPU speedup: {speedup:.2f}x faster")
        except:
            pass
        
        print("\n✓ GPU is working correctly!")
        print("="*80 + "\n")
        return True
        
    except Exception as e:
        print(f"\n✗ GPU test failed: {e}")
        print("="*80 + "\n")
        return False


def get_gpu_memory_info():
    """Get current GPU memory usage"""
    try:
        from tensorflow.python.client import device_lib
        
        local_devices = device_lib.list_local_devices()
        gpu_devices = [d for d in local_devices if d.device_type == 'GPU']
        
        if not gpu_devices:
            return None
        
        memory_info = []
        for device in gpu_devices:
            info = {
                'name': device.name,
                'memory_limit': device.memory_limit / (1024**3),  # Convert to GB
            }
            memory_info.append(info)
        
        return memory_info
    except:
        return None


def main():
    """Test GPU utilities"""
    print("\n" + "="*80)
    print("GPU UTILITIES - DIAGNOSTICS")
    print("="*80 + "\n")
    
    # Print GPU info
    print_gpu_info()
    
    # Configure GPU
    gpu_available = configure_gpu(memory_growth=True)
    
    # Test GPU if available
    if gpu_available:
        test_gpu()
    
    print("\nDiagnostics complete!")


if __name__ == '__main__':
    main()

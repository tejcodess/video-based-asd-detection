from __future__ import print_function
from keras import backend as K

# Make sure that you have tensorflow-gpu installed if you want o use init_devices('gpu')


def init_devices(device_type=None):
    if device_type is None:
        device_type = 'cpu'

    num_cores = 6

    if device_type == 'gpu':
        num_GPU = 1
        num_CPU = 1
    else:
        num_CPU = 1
        num_GPU = 0

    # Note: TensorFlow 2.x handles device placement automatically
    # This function is kept for backwards compatibility but may not work with TF 2.x
    print(f"Device configuration: {device_type} - CPU: {num_CPU}, GPU: {num_GPU}")


def print_devices():
    import tensorflow as tf
    print("Available devices:")
    for device in tf.config.list_physical_devices():
        print(f"  {device}")

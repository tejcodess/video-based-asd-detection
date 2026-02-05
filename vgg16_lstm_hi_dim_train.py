import numpy as np
from keras import backend as K
import os
import sys
from config import TRAINING_DATA_PATH, MODELS_PATH, REPORTS_PATH
from gpu_utils import configure_gpu, print_gpu_info


def main():
    K.set_image_data_format('channels_last')
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from plot_utils import plot_and_save_history
    from recurrent_networks import vgg16LSTMVideoClassifier
    from UCF101_loader import load_ucf
    
    # Configure GPU
    print_gpu_info()
    configure_gpu(memory_growth=True)

    print(f"\nTraining Configuration:")
    print(f"  Data path: {TRAINING_DATA_PATH}")
    print(f"  Models path: {MODELS_PATH}")
    print(f"  Reports path: {REPORTS_PATH}")

    np.random.seed(42)

    # this line downloads the video files of UCF-101 dataset if they are not available in the very_large_data folder
    load_ucf(os.path.dirname(TRAINING_DATA_PATH))

    classifier = vgg16LSTMVideoClassifier()

    history = classifier.fit(data_dir_path=TRAINING_DATA_PATH, model_dir_path=MODELS_PATH, vgg16_include_top=False,
                             data_set_name='training_set')

    plot_and_save_history(history, vgg16LSTMVideoClassifier.model_name,
                          os.path.join(REPORTS_PATH, vgg16LSTMVideoClassifier.model_name + '-hi-dim-history.png'))
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()

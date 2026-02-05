import numpy as np
from keras import backend as K
import os
from config import TRAINING_DATA_PATH, MODELS_PATH, REPORTS_PATH
from plot_utils import plot_and_save_history
from recurrent_networks import vgg16LSTMVideoClassifier
from UCF101_loader import load_ucf
from gpu_utils import configure_gpu, print_gpu_info

K.set_image_data_format('channels_last')
np.random.seed(42)

def main():
    print("=" * 80)
    print("VGG16-LSTM Video Classifier Training")
    print("=" * 80)
    
    # Configure GPU for training
    print_gpu_info()
    gpu_available = configure_gpu(memory_growth=True)
    
    if gpu_available:
        print("✓ Training will use GPU acceleration\n")
    else:
        print("⚠ Training will use CPU (this will be slower)\n")
    print(f"\nTraining Configuration:")
    print(f"  Data path: {TRAINING_DATA_PATH}")
    print(f"  Models path: {MODELS_PATH}")
    print(f"  Reports path: {REPORTS_PATH}")
    print()
    
    # Verify data directory exists
    if not os.path.exists(TRAINING_DATA_PATH):
        print(f"ERROR: Training data path does not exist: {TRAINING_DATA_PATH}")
        print("Please ensure your dataset is set up correctly.")
        return
    
    # List the classes found
    classes = [d for d in os.listdir(TRAINING_DATA_PATH) 
               if os.path.isdir(os.path.join(TRAINING_DATA_PATH, d))]
    print(f"Found {len(classes)} classes: {classes}")
    
    # Count videos per class
    for cls in classes:
        cls_path = os.path.join(TRAINING_DATA_PATH, cls)
        video_count = len([f for f in os.listdir(cls_path) 
                          if os.path.isfile(os.path.join(cls_path, f))])
        print(f"  {cls}: {video_count} videos")
    
    print("\nLoading dataset...")
    load_ucf(os.path.dirname(TRAINING_DATA_PATH))
    
    print("\nInitializing classifier...")
    classifier = vgg16LSTMVideoClassifier()
    
    print("\nStarting training (this may take a while)...")
    print("-" * 80)
    history = classifier.fit(
        data_dir_path=TRAINING_DATA_PATH,
        model_dir_path=MODELS_PATH,
        vgg16_include_top=False,
        data_set_name='training_set'
    )
    
    print("\nSaving training history plot...")
    plot_file = os.path.join(REPORTS_PATH, f'{vgg16LSTMVideoClassifier.model_name}-history.png')
    plot_and_save_history(
        history,
        vgg16LSTMVideoClassifier.model_name,
        plot_file
    )
    print(f"Plot saved to: {plot_file}")
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Models saved to: {MODELS_PATH}")
    print("=" * 80)

if __name__ == '__main__':
    main()

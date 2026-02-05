import numpy as np
from keras import backend as K
import sys
import os
import gc
from config import TESTING_DATA_PATH, MODELS_PATH, CLASSES
from recurrent_networks import vgg16LSTMVideoClassifier
from UCF101_loader import load_ucf, scan_ucf_with_labels
from gpu_utils import configure_gpu

K.set_image_data_format('channels_last')
np.random.seed(42)

def main():
    print("=" * 80)
    print("VGG16-LSTM Video Classifier Prediction")
    print("=" * 80)
    
    # Configure GPU for prediction
    configure_gpu(memory_growth=True)
    print(f"\nPrediction Configuration:")
    print(f"  Test data path: {TESTING_DATA_PATH}")
    print(f"  Models path: {MODELS_PATH}")
    print(f"  Classes: {CLASSES}")
    print()
    
    # Verify test data directory exists
    if not os.path.exists(TESTING_DATA_PATH):
        print(f"ERROR: Testing data path does not exist: {TESTING_DATA_PATH}")
        print("Please ensure your dataset is set up correctly.")
        return
    
    # List the classes found
    classes = [d for d in os.listdir(TESTING_DATA_PATH) 
               if os.path.isdir(os.path.join(TESTING_DATA_PATH, d))]
    print(f"Found {len(classes)} test classes: {classes}")
    
    # Count videos per class
    total_videos = 0
    for cls in classes:
        cls_path = os.path.join(TESTING_DATA_PATH, cls)
        video_count = len([f for f in os.listdir(cls_path) 
                          if os.path.isfile(os.path.join(cls_path, f))])
        print(f"  {cls}: {video_count} videos")
        total_videos += video_count
    
    vgg16_include_top = False
    config_file_path = vgg16LSTMVideoClassifier.get_config_file_path(
        MODELS_PATH, vgg16_include_top=vgg16_include_top)
    weight_file_path = vgg16LSTMVideoClassifier.get_weight_file_path(
        MODELS_PATH, vgg16_include_top=vgg16_include_top)
    
    # Check if model files exist
    if not os.path.exists(config_file_path):
        print(f"\nERROR: Model config file not found: {config_file_path}")
        print("Please train the model first using train_simple.py")
        return
    
    if not os.path.exists(weight_file_path):
        print(f"\nERROR: Model weights file not found: {weight_file_path}")
        print("Please train the model first using train_simple.py")
        return
    
    print(f"\nLoading model from:")
    print(f"  Config: {config_file_path}")
    print(f"  Weights: {weight_file_path}")
    
    load_ucf(os.path.dirname(TESTING_DATA_PATH))
    
    predictor = vgg16LSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)
    
    print(f"\nModel loaded. Available labels: {list(predictor.labels.keys())}")
    
    videos = scan_ucf_with_labels(
        TESTING_DATA_PATH, 
        [label for (label, label_index) in predictor.labels.items()]
    )
    
    if len(videos) == 0:
        print("\nERROR: No videos found to predict!")
        return
    
    video_file_path_list = np.array([file_path for file_path in videos.keys()])
    np.random.shuffle(video_file_path_list)
    
    correct_count = 0
    count = 0
    predictions_by_class = {cls: {'correct': 0, 'total': 0} for cls in CLASSES}
    
    print(f"\n{'=' * 80}")
    print(f"Processing {len(video_file_path_list)} videos...")
    print(f"{'=' * 80}\n")
    
    for video_file_path in video_file_path_list:
        label = videos[video_file_path]
        predicted_label = predictor.predict(video_file_path)
        
        is_correct = (label == predicted_label)
        status = "✓" if is_correct else "✗"
        
        print(f"{status} Video: {os.path.basename(video_file_path)[:50]}")
        print(f"  Predicted: {predicted_label} | Actual: {label}")
        
        if is_correct:
            correct_count += 1
            predictions_by_class[label]['correct'] += 1
        
        predictions_by_class[label]['total'] += 1
        count += 1
        
        current_accuracy = correct_count / count
        print(f"  Running Accuracy: {current_accuracy:.4f} ({correct_count}/{count})")
        print()
        
        gc.collect()
    
    # Final statistics
    print("=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(f"\nOverall Accuracy: {correct_count}/{count} = {correct_count/count:.4f} ({correct_count/count*100:.2f}%)")
    
    print("\nPer-Class Performance:")
    for cls in CLASSES:
        if predictions_by_class[cls]['total'] > 0:
            cls_correct = predictions_by_class[cls]['correct']
            cls_total = predictions_by_class[cls]['total']
            cls_accuracy = cls_correct / cls_total
            print(f"  {cls}: {cls_correct}/{cls_total} = {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)")
        else:
            print(f"  {cls}: No videos tested")
    
    print("=" * 80)
    
    # Optionally save results to CSV
    output_file = os.path.join(MODELS_PATH, "prediction_results.txt")
    with open(output_file, 'w') as f:
        f.write(f"Overall Accuracy: {correct_count/count:.4f}\n")
        f.write(f"Correct: {correct_count}/{count}\n\n")
        f.write("Per-Class Performance:\n")
        for cls in CLASSES:
            if predictions_by_class[cls]['total'] > 0:
                cls_correct = predictions_by_class[cls]['correct']
                cls_total = predictions_by_class[cls]['total']
                cls_accuracy = cls_correct / cls_total
                f.write(f"  {cls}: {cls_correct}/{cls_total} = {cls_accuracy:.4f}\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()

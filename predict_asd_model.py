"""
Updated Prediction Script for ASD Video Classification
Adapted for Windows environment with Python 3.9+ compatibility
Dataset: D:/projects/01/dataset/autism_data_anonymized
"""

import numpy as np
import os
import sys
from pathlib import Path
import csv
from datetime import datetime

# Set image format for Keras backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from recurrent_networks import vgg16LSTMVideoClassifier
from asd_data_loader import scan_dataset_with_labels, scan_dataset

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SETUP
# ============================================================================

# Your test dataset location
TEST_DATASET_PATH = r"D:\projects\01\dataset\autism_data_anonymized\testing_set"
DATA_SET_NAME = 'autism_data'

# Model directory (where trained model is saved)
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / 'models' / DATA_SET_NAME

# Output CSV file for predictions
OUTPUT_CSV = PROJECT_ROOT / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

# Model configuration (should match training)
VGG16_INCLUDE_TOP = False  # Use high-dimensional features (recommended)

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def check_model_files():
    """Verify trained model files exist"""
    config_file = vgg16LSTMVideoClassifier.get_config_file_path(
        str(MODEL_DIR), 
        vgg16_include_top=VGG16_INCLUDE_TOP
    )
    weight_file = vgg16LSTMVideoClassifier.get_weight_file_path(
        str(MODEL_DIR), 
        vgg16_include_top=VGG16_INCLUDE_TOP
    )
    
    if not Path(config_file).exists():
        raise FileNotFoundError(
            f"Model config file not found: {config_file}\n"
            f"Please train the model first using train_asd_model.py"
        )
    
    if not Path(weight_file).exists():
        raise FileNotFoundError(
            f"Model weights file not found: {weight_file}\n"
            f"Please train the model first using train_asd_model.py"
        )
    
    print(f"✓ Model config: {Path(config_file).name}")
    print(f"✓ Model weights: {Path(weight_file).name}")
    
    return config_file, weight_file


def verify_test_dataset():
    """Verify test dataset exists and has correct structure"""
    test_path = Path(TEST_DATASET_PATH)
    
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found at: {test_path}\n"
            f"Please update TEST_DATASET_PATH in the script."
        )
    
    print(f"\n{'='*70}")
    print(f"Test Dataset Verification")
    print(f"{'='*70}")
    print(f"Dataset location: {test_path}")
    
    # Check for subdirectories (classes)
    classes = [d for d in test_path.iterdir() if d.is_dir()]
    
    if len(classes) == 0:
        raise ValueError(f"No class folders found in {test_path}")
    
    print(f"\nFound {len(classes)} classes:")
    total_videos = 0
    
    for class_dir in sorted(classes):
        video_files = (
            list(class_dir.glob('*.avi')) + 
            list(class_dir.glob('*.mp4')) +
            list(class_dir.glob('*.mov')) +
            list(class_dir.glob('*.mkv'))
        )
        video_count = len(video_files)
        total_videos += video_count
        print(f"  - {class_dir.name}: {video_count} videos")
    
    print(f"\nTotal test videos: {total_videos}")
    
    if total_videos == 0:
        raise ValueError(f"No video files found in class folders")
    
    print(f"{'='*70}\n")
    return classes, total_videos


def predict_with_aggregation(video_file_path, predictor, video_name, true_label, csv_writer):
    """
    Predict class for a single video and write to CSV.
    
    Args:
        video_file_path: Path to video file
        predictor: Trained classifier
        video_name: Name of the video file
        true_label: Actual label (ground truth)
        csv_writer: CSV writer object for output
        
    Returns:
        tuple: (predicted_label, confidence, is_correct)
    """
    try:
        # Get prediction and confidence
        predicted_c = predictor.predict_with_confidence(video_file_path)
        predicted_label = predictor.labels_idx2word[np.argmax(predicted_c)]
        confidence = np.max(predicted_c)
        
        is_correct = (predicted_label == true_label)
        
        # Write to CSV
        csv_writer.writerow([
            video_name,
            true_label,
            predicted_label,
            is_correct,
            f"{confidence:.4f}",
            f"{predicted_c[0]:.4f}" if len(predicted_c) > 0 else "",
            f"{predicted_c[1]:.4f}" if len(predicted_c) > 1 else ""
        ])
        
        return predicted_label, confidence, is_correct
        
    except Exception as e:
        print(f"  ✗ Error predicting {video_name}: {e}")
        csv_writer.writerow([
            video_name,
            true_label,
            "ERROR",
            False,
            "0.0",
            "",
            ""
        ])
        return None, 0.0, False


def main():
    """Main prediction function"""
    print("\n" + "="*70)
    print("ASD Video Classification - Prediction Script")
    print("="*70)
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print("="*70 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Step 1: Check model files
    print("\n" + "="*70)
    print("Checking Trained Model")
    print("="*70)
    
    try:
        config_file, weight_file = check_model_files()
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        return
    
    # Step 2: Verify test dataset
    try:
        classes, total_videos = verify_test_dataset()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ {e}")
        return
    
    # Step 3: Load the trained model
    print("\n" + "="*70)
    print("Loading Trained Model")
    print("="*70)
    
    predictor = vgg16LSTMVideoClassifier()
    
    try:
        predictor.load_model(config_file, weight_file)
        print(f"✓ Model loaded successfully")
        print(f"  Classes: {list(predictor.labels.keys())}")
        print(f"  Expected frames: {predictor.expected_frames}")
        print(f"  Input tokens: {predictor.num_input_tokens}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    print("="*70 + "\n")
    
    # Step 4: Get all test videos
    print("\n" + "="*70)
    print("Scanning Test Videos")
    print("="*70)
    
    try:
        videos = scan_dataset_with_labels(
            TEST_DATASET_PATH, 
            [label for label in predictor.labels.keys()]
        )
    except Exception as e:
        print(f"✗ Error scanning videos: {e}")
        return
    
    video_file_paths = list(videos.keys())
    np.random.shuffle(video_file_paths)
    
    # Step 5: Run predictions and save to CSV
    print("\n" + "="*70)
    print("Running Predictions")
    print("="*70)
    print(f"Output CSV: {OUTPUT_CSV}")
    print("="*70 + "\n")
    
    # Create CSV file
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        csv_writer.writerow([
            'Video Name',
            'True Label',
            'Predicted Label',
            'Correct',
            'Confidence',
            f'Class {list(predictor.labels.keys())[0]} Prob',
            f'Class {list(predictor.labels.keys())[1]} Prob' if len(predictor.labels) > 1 else ''
        ])
        
        correct_count = 0
        total_count = 0
        predictions_by_class = {label: {'correct': 0, 'total': 0} for label in predictor.labels.keys()}
        
        # Process each video
        for i, video_file_path in enumerate(video_file_paths, 1):
            video_name = Path(video_file_path).name
            true_label = videos[video_file_path]
            
            print(f"[{i}/{len(video_file_paths)}] Processing: {video_name}")
            
            predicted_label, confidence, is_correct = predict_with_aggregation(
                video_file_path, predictor, video_name, true_label, csv_writer
            )
            
            if predicted_label:
                print(f"  True: {true_label} | Predicted: {predicted_label} | "
                      f"Confidence: {confidence:.4f} | {'✓' if is_correct else '✗'}")
                
                if is_correct:
                    correct_count += 1
                
                total_count += 1
                predictions_by_class[true_label]['total'] += 1
                
                if is_correct:
                    predictions_by_class[true_label]['correct'] += 1
            
            # Show running accuracy
            if total_count > 0:
                running_accuracy = correct_count / total_count
                print(f"  Running accuracy: {running_accuracy:.4f} ({correct_count}/{total_count})")
            print()
    
    # Step 6: Display final results
    print("\n" + "="*70)
    print("Prediction Results")
    print("="*70)
    
    if total_count > 0:
        overall_accuracy = correct_count / total_count
        print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({correct_count}/{total_count})")
        print(f"Overall Error Rate: {1 - overall_accuracy:.4f}")
        
        print(f"\nPer-Class Results:")
        for label in sorted(predictor.labels.keys()):
            stats = predictions_by_class[label]
            if stats['total'] > 0:
                class_acc = stats['correct'] / stats['total']
                print(f"  {label}:")
                print(f"    Accuracy: {class_acc:.4f} ({stats['correct']}/{stats['total']})")
            else:
                print(f"  {label}: No test samples")
        
        print(f"\nPredictions saved to: {OUTPUT_CSV}")
        print(f"\nYou can use this CSV file for:")
        print(f"  1. Analyzing individual video predictions")
        print(f"  2. Computing aggregated predictions (e.g., confidence threshold)")
        print(f"  3. Generating confusion matrices and other metrics")
    else:
        print("\n✗ No predictions were made. Check for errors above.")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

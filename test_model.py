"""
Test Trained Model on Testing Dataset
Evaluates model performance on unseen testing videos
"""

import numpy as np
from keras import backend as K
import os
from pathlib import Path

# GPU configuration
try:
    from gpu_utils import configure_gpu
    print("="*70)
    print("INITIALIZING GPU...")
    print("="*70)
    gpu_configured = configure_gpu(memory_growth=True)
    if gpu_configured:
        print("✓ GPU configured successfully")
    else:
        print("⚠ No GPU detected, using CPU")
    print("="*70)
except Exception as e:
    print(f"⚠ GPU configuration failed: {e}")
    print("Continuing with CPU...")

from config import TESTING_DATA_PATH, MODELS_PATH, USE_MINI_DATASET
from recurrent_networks import vgg16LSTMVideoClassifier
from vgg16_feature_extractor import extract_vgg16_features_live

K.set_image_data_format('channels_last')

def test_model():
    """Test the trained model on testing dataset"""
    
    print("\n" + "="*70)
    print("TESTING TRAINED MODEL")
    print("="*70)
    
    if USE_MINI_DATASET:
        print("✓ Using mini dataset (testing mode)")
    else:
        print("✓ Using full dataset")
    
    print(f"\nTesting data: {TESTING_DATA_PATH}")
    print(f"Model location: {MODELS_PATH}")
    print("="*70)
    
    # Check if model exists
    model_config = os.path.join(MODELS_PATH, 'vgg16-lstm-hi-dim-config.npy')
    model_weights = os.path.join(MODELS_PATH, 'vgg16-lstm-hi-dim-weights.h5')
    
    if not os.path.exists(model_config) or not os.path.exists(model_weights):
        print("\n" + "!"*70)
        print("ERROR: Trained model not found!")
        print("!"*70)
        print("Please train the model first by running:")
        print("  python quick_test.py")
        print("!"*70)
        return
    
    # Load the model
    print("\n[STEP 1/4] Loading trained model...")
    print("-"*70)
    classifier = vgg16LSTMVideoClassifier()
    classifier.load_model(
        config_file_path=model_config,
        weight_file_path=model_weights
    )
    print(f"✓ Model loaded")
    print(f"  Classes: {classifier.labels}")
    print(f"  Expected frames: {classifier.expected_frames}")
    
    # Get testing videos
    print("\n[STEP 2/4] Scanning testing dataset...")
    print("-"*70)
    
    test_videos = {}
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    
    for class_name in ['ASD', 'TD']:
        class_dir = os.path.join(TESTING_DATA_PATH, class_name)
        if not os.path.exists(class_dir):
            print(f"  Warning: {class_name} directory not found")
            continue
        
        videos = []
        for ext in video_extensions:
            videos.extend(list(Path(class_dir).glob(f'*{ext}')))
        
        test_videos[class_name] = videos
        print(f"  {class_name}: {len(videos)} videos")
    
    total_test_videos = sum(len(v) for v in test_videos.values())
    print(f"\nTotal testing videos: {total_test_videos}")
    
    if total_test_videos == 0:
        print("\n" + "!"*70)
        print("ERROR: No testing videos found!")
        print("!"*70)
        return
    
    # Test each video
    print("\n[STEP 3/4] Running predictions...")
    print("-"*70)
    
    results = []
    correct = 0
    total = 0
    
    for true_label, videos in test_videos.items():
        for video_path in videos:
            video_name = video_path.name
            
            # Predict
            try:
                predicted_label = classifier.predict(str(video_path))
                is_correct = (predicted_label == true_label)
                
                if is_correct:
                    correct += 1
                total += 1
                
                status = "✓" if is_correct else "✗"
                print(f"  {status} {video_name}")
                print(f"      True: {true_label}, Predicted: {predicted_label}")
                
                results.append({
                    'video': video_name,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'correct': is_correct
                })
                
            except Exception as e:
                print(f"  ✗ {video_name}: Error - {e}")
                total += 1
    
    # Calculate accuracy
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Print results
    print("\n[STEP 4/4] Results Summary")
    print("-"*70)
    print(f"\nTotal videos tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    print("\nPer-Class Results:")
    for class_name in test_videos.keys():
        class_results = [r for r in results if r['true_label'] == class_name]
        if len(class_results) > 0:
            class_correct = sum(1 for r in class_results if r['correct'])
            class_accuracy = (class_correct / len(class_results) * 100)
            print(f"  {class_name}: {class_correct}/{len(class_results)} correct ({class_accuracy:.2f}%)")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("              ASD    TD")
    print("       ASD     ", end="")
    asd_to_asd = sum(1 for r in results if r['true_label'] == 'ASD' and r['predicted_label'] == 'ASD')
    asd_to_td = sum(1 for r in results if r['true_label'] == 'ASD' and r['predicted_label'] == 'TD')
    print(f"{asd_to_asd:2d}     {asd_to_td:2d}")
    
    print("Actual  TD     ", end="")
    td_to_asd = sum(1 for r in results if r['true_label'] == 'TD' and r['predicted_label'] == 'ASD')
    td_to_td = sum(1 for r in results if r['true_label'] == 'TD' and r['predicted_label'] == 'TD')
    print(f"{td_to_asd:2d}     {td_to_td:2d}")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    
    # Interpretation
    print("\nInterpretation:")
    if USE_MINI_DATASET:
        print("⚠ Note: This is a very small test set (6 videos)")
        print("  Results may not be statistically significant")
        print("  The model was trained on only 8 videos (overfitting expected)")
        print("\nFor reliable results:")
        print("  1. Set USE_MINI_DATASET = False in config.py")
        print("  2. Run: python train_simple.py")
        print("  3. Run: python test_model.py")
    else:
        if accuracy >= 80:
            print("✓ Excellent performance!")
        elif accuracy >= 70:
            print("✓ Good performance")
        elif accuracy >= 60:
            print("⚠ Moderate performance - consider more training")
        else:
            print("⚠ Low performance - model may need retraining or different architecture")
    
    print("\n")

if __name__ == '__main__':
    try:
        test_model()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
    except Exception as e:
        print("\n" + "!"*70)
        print("ERROR OCCURRED:")
        print("!"*70)
        print(f"{e}")
        import traceback
        traceback.print_exc()
        print("!"*70)

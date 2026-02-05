"""
Dataset Verification and Structure Analysis Script
Verifies your dataset structure and provides recommendations
"""

import os
from pathlib import Path
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAINING_DATA_PATH = r"D:\projects\01\dataset\autism_data_anonymized\training_set"
TESTING_DATA_PATH = r"D:\projects\01\dataset\autism_data_anonymized\testing_set"

# Supported video formats
VIDEO_EXTENSIONS = ['.avi', '.mp4', '.mov', '.mkv']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def analyze_directory(dir_path, label="Directory"):
    """Analyze a directory for video files"""
    path = Path(dir_path)
    
    if not path.exists():
        print(f"  ✗ {label} NOT FOUND: {dir_path}")
        return None
    
    print(f"\n  ✓ {label}: {dir_path}")
    
    # Get class directories
    class_dirs = [d for d in path.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        print(f"    ⚠ WARNING: No class folders found")
        return None
    
    print(f"    Classes found: {len(class_dirs)}")
    
    total_videos = 0
    class_info = {}
    
    for class_dir in sorted(class_dirs):
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(list(class_dir.glob(f'*{ext}')))
        
        video_count = len(video_files)
        total_videos += video_count
        class_info[class_dir.name] = video_count
        
        # Sample some video names
        sample_videos = [v.name for v in video_files[:3]]
        
        print(f"\n    Class: {class_dir.name}")
        print(f"      Videos: {video_count}")
        if sample_videos:
            print(f"      Sample files:")
            for sample in sample_videos:
                print(f"        - {sample}")
    
    print(f"\n    Total videos in {label}: {total_videos}")
    
    return {
        'path': str(path),
        'classes': class_info,
        'total_videos': total_videos
    }


def check_video_file(video_path):
    """Check if a video file can be opened"""
    try:
        import cv2
        vidcap = cv2.VideoCapture(str(video_path))
        if not vidcap.isOpened():
            return False, "Cannot open video file"
        
        # Try to read first frame
        success, frame = vidcap.read()
        if not success:
            return False, "Cannot read video frames"
        
        # Get video properties
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        vidcap.release()
        
        return True, {
            'fps': fps,
            'frames': frame_count,
            'resolution': f"{width}x{height}"
        }
    except Exception as e:
        return False, str(e)


def verify_prerequisites():
    """Check if required libraries are installed"""
    print("\n" + "="*70)
    print("Checking Prerequisites")
    print("="*70)
    
    issues = []
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        issues.append("Python version should be 3.9 or higher")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow: {tf.__version__}")
    except ImportError:
        issues.append("TensorFlow not installed")
        print("✗ TensorFlow: NOT INSTALLED")
    
    # Check Keras
    try:
        from tensorflow import keras
        print(f"✓ Keras: {keras.__version__}")
    except ImportError:
        issues.append("Keras not available")
        print("✗ Keras: NOT AVAILABLE")
    
    # Check OpenCV
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        issues.append("OpenCV not installed")
        print("✗ OpenCV: NOT INSTALLED")
    
    # Check NumPy
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        issues.append("NumPy not installed")
        print("✗ NumPy: NOT INSTALLED")
    
    # Check scikit-learn
    try:
        import sklearn
        print(f"✓ scikit-learn: {sklearn.__version__}")
    except ImportError:
        issues.append("scikit-learn not installed")
        print("✗ scikit-learn: NOT INSTALLED")
    
    if issues:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ All prerequisites satisfied")
        return True


def main():
    """Main verification function"""
    print("\n" + "="*70)
    print("ASD Video Dataset Verification")
    print("="*70)
    
    # Check prerequisites
    prereqs_ok = verify_prerequisites()
    
    # Check training data
    print("\n" + "="*70)
    print("Training Dataset Analysis")
    print("="*70)
    training_info = analyze_directory(TRAINING_DATA_PATH, "Training Data")
    
    # Check testing data
    print("\n" + "="*70)
    print("Testing Dataset Analysis")
    print("="*70)
    testing_info = analyze_directory(TESTING_DATA_PATH, "Testing Data")
    
    # Sample video check (if OpenCV is available)
    if prereqs_ok:
        print("\n" + "="*70)
        print("Sample Video Check")
        print("="*70)
        
        # Try to find a sample video
        training_path = Path(TRAINING_DATA_PATH)
        if training_path.exists():
            sample_video = None
            for class_dir in training_path.iterdir():
                if class_dir.is_dir():
                    for ext in VIDEO_EXTENSIONS:
                        videos = list(class_dir.glob(f'*{ext}'))
                        if videos:
                            sample_video = videos[0]
                            break
                if sample_video:
                    break
            
            if sample_video:
                print(f"\nTesting sample video: {sample_video.name}")
                success, result = check_video_file(sample_video)
                
                if success:
                    print(f"  ✓ Video is readable")
                    print(f"    FPS: {result['fps']}")
                    print(f"    Frames: {result['frames']}")
                    print(f"    Resolution: {result['resolution']}")
                    duration = result['frames'] / result['fps'] if result['fps'] > 0 else 0
                    print(f"    Duration: {duration:.2f} seconds")
                else:
                    print(f"  ✗ Error: {result}")
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("Summary and Recommendations")
    print("="*70)
    
    if training_info and testing_info:
        print("\n✓ Dataset structure looks good!")
        print(f"\nTraining: {training_info['total_videos']} videos across {len(training_info['classes'])} classes")
        print(f"Testing: {testing_info['total_videos']} videos across {len(testing_info['classes'])} classes")
        
        # Check class balance
        print("\nClass Distribution:")
        print("  Training:")
        for class_name, count in training_info['classes'].items():
            print(f"    {class_name}: {count} videos")
        
        print("  Testing:")
        for class_name, count in testing_info['classes'].items():
            print(f"    {class_name}: {count} videos")
        
        # Recommendations
        print("\nRecommendations:")
        
        # Check if classes match
        train_classes = set(training_info['classes'].keys())
        test_classes = set(testing_info['classes'].keys())
        
        if train_classes != test_classes:
            print("  ⚠ Warning: Training and testing classes don't match!")
            print(f"    Training: {sorted(train_classes)}")
            print(f"    Testing: {sorted(test_classes)}")
        else:
            print(f"  ✓ Class names match between training and testing")
        
        # Check class balance
        train_counts = list(training_info['classes'].values())
        if max(train_counts) / min(train_counts) > 2:
            print("  ⚠ Warning: Classes are imbalanced (ratio > 2:1)")
            print("    Consider using class weights during training")
        else:
            print("  ✓ Classes are reasonably balanced")
        
        # Check dataset size
        if training_info['total_videos'] < 50:
            print(f"  ⚠ Warning: Small training dataset ({training_info['total_videos']} videos)")
            print("    Consider data augmentation or reducing model complexity")
        elif training_info['total_videos'] < 100:
            print(f"  ⚠ Note: Moderate training dataset ({training_info['total_videos']} videos)")
            print("    Model performance may benefit from more data")
        else:
            print(f"  ✓ Good training dataset size ({training_info['total_videos']} videos)")
        
        print("\n✓ You're ready to start training!")
        print("  Run: python train_asd_model.py")
        
    else:
        print("\n✗ Dataset issues detected!")
        if not training_info:
            print("  - Training data not found or has issues")
        if not testing_info:
            print("  - Testing data not found or has issues")
        print("\nPlease fix these issues before training.")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

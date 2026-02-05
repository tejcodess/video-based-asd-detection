"""
Updated VGG16 Feature Extractor for ASD Dataset
Works with custom dataset structure (training/ and testing/ subdirectories)
Python 3.9+ compatible
"""

import cv2
import os
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
import gc
from pathlib import Path

MAX_NB_CLASSES = 2


def extract_vgg16_features_live(model, video_input_file_path):
    """
    Extract VGG16 features from a video file in real-time (no caching).
    
    Args:
        model: VGG16 model for feature extraction
        video_input_file_path: Path to input video file
        
    Returns:
        numpy array of extracted features
    """
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    
    if not vidcap.isOpened():
        raise ValueError(f"Cannot open video file: {video_input_file_path}")
    
    features = []
    count = 0
    
    while True:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # Extract 1 frame per second
        success, image = vidcap.read()
        
        if not success:
            break
            
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input_array = img_to_array(img)
        input_array = np.expand_dims(input_array, axis=0)
        input_array = preprocess_input(input_array)
        
        feature = model.predict(input_array, verbose=0).ravel()
        features.append(feature)
        count += 1
        gc.collect()
    
    vidcap.release()
    unscaled_features = np.array(features)
    
    return unscaled_features


def extract_vgg16_features(model, video_input_file_path, feature_output_file_path):
    """
    Extract VGG16 features from a video file with caching.
    
    Args:
        model: VGG16 model for feature extraction
        video_input_file_path: Path to input video file
        feature_output_file_path: Path to save extracted features
        
    Returns:
        numpy array of extracted features
    """
    # Check if features already extracted
    if os.path.exists(feature_output_file_path):
        print(f'Loading cached features: {Path(feature_output_file_path).name}')
        return np.load(feature_output_file_path)
    
    print('Extracting frames from video: ', Path(video_input_file_path).name)
    vidcap = cv2.VideoCapture(video_input_file_path)
    
    if not vidcap.isOpened():
        raise ValueError(f"Cannot open video file: {video_input_file_path}")
    
    features = []
    count = 0
    
    while True:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # Extract 1 frame per second
        success, image = vidcap.read()
        
        if not success:
            break
            
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input_array = img_to_array(img)
        input_array = np.expand_dims(input_array, axis=0)
        input_array = preprocess_input(input_array)
        
        feature = model.predict(input_array, verbose=0).ravel()
        features.append(feature)
        count += 1
        gc.collect()
    
    vidcap.release()
    unscaled_features = np.array(features)
    
    # Save features for future use
    np.save(feature_output_file_path, unscaled_features)
    print(f'  Extracted {count} frames, shape: {unscaled_features.shape}')
    
    return unscaled_features


def scan_and_extract_vgg16_features(data_dir_path, output_dir_path, model=None, data_set_name=None):
    """
    Scan dataset directory and extract VGG16 features from all videos.
    Compatible with custom dataset structure.
    
    Args:
        data_dir_path: Path to dataset directory (should be the training/ or testing/ folder)
        output_dir_path: Directory name for saving features (created inside data_dir_path parent)
        model: VGG16 model (created if None)
        data_set_name: Dataset name (not used in new structure, kept for compatibility)
        
    Returns:
        tuple: (x_samples, y_samples) where x_samples are features and y_samples are labels
    """
    # data_dir_path should point directly to training/ or testing/ folder
    input_data_dir_path = Path(data_dir_path)
    
    # Create output directory in the same parent as input
    if data_dir_path.endswith('training'):
        base_dir = input_data_dir_path.parent
    else:
        base_dir = input_data_dir_path.parent if input_data_dir_path.parent.name != 'very_large_data' else input_data_dir_path
    
    output_feature_data_dir_path = base_dir / output_dir_path
    
    if not input_data_dir_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {input_data_dir_path}")
    
    # Create VGG16 model if not provided
    if model is None:
        print("Creating VGG16 model...")
        model = VGG16(include_top=True, weights='imagenet')
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Create output directory
    output_feature_data_dir_path.mkdir(parents=True, exist_ok=True)
    
    y_samples = []
    x_samples = []
    
    # Get all class directories
    class_dirs = [d for d in input_data_dir_path.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {input_data_dir_path}")
    
    print(f"\n{'='*70}")
    print(f"Extracting VGG16 Features")
    print(f"{'='*70}")
    print(f"Dataset: {input_data_dir_path}")
    print(f"Classes: {[d.name for d in class_dirs]}")
    print(f"Output: {output_feature_data_dir_path}")
    print(f"{'='*70}\n")
    
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    dir_count = 0
    total_videos = 0
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        output_class_dir = output_feature_data_dir_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(class_dir.glob(f'*{ext}')))
        
        print(f"\nProcessing class '{class_name}': {len(video_files)} videos")
        
        for video_file in sorted(video_files):
            video_file_path = str(video_file)
            output_feature_file_path = output_class_dir / f"{video_file.stem}.npy"
            
            try:
                x = extract_vgg16_features(model, video_file_path, str(output_feature_file_path))
                y_samples.append(class_name)
                x_samples.append(x)
                total_videos += 1
            except Exception as e:
                print(f"  ✗ Error processing {video_file.name}: {e}")
                continue
            
            gc.collect()
        
        dir_count += 1
        
        # Remove the MAX_NB_CLASSES limit for custom datasets
        # if dir_count >= MAX_NB_CLASSES:
        #     break
    
    print(f"\n{'='*70}")
    print(f"Feature Extraction Complete")
    print(f"{'='*70}")
    print(f"Total videos processed: {total_videos}")
    print(f"Classes: {len(set(y_samples))}")
    print(f"Features saved to: {output_feature_data_dir_path}")
    print(f"{'='*70}\n")
    
    if len(x_samples) == 0:
        raise ValueError("No features extracted. Check your video files and paths.")
    
    return x_samples, y_samples


def main():
    """Test feature extraction"""
    # Example usage
    data_dir_path = r"D:\projects\01\dataset\autism_data_anonymized\training"
    output_dir_path = "autism_data-vgg16-HiDimFeatures"
    
    print("="*70)
    print("Testing VGG16 Feature Extractor")
    print("="*70)
    
    try:
        # Create VGG16 model
        model = VGG16(include_top=False, weights='imagenet')
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Extract features
        x_samples, y_samples = scan_and_extract_vgg16_features(
            data_dir_path, 
            output_dir_path, 
            model=model
        )
        
        print("\nExtraction successful!")
        print(f"Samples: {len(x_samples)}")
        print(f"Labels: {set(y_samples)}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease update data_dir_path to point to your training folder.")


if __name__ == '__main__':
    main()

"""
Custom Data Loader for ASD Video Dataset
Replaces UCF101_loader with support for custom dataset structure
Compatible with training/ and testing/ subdirectories
"""

import os
import sys
from pathlib import Path


def scan_dataset(data_dir_path, limit=None):
    """
    Scan dataset directory for video files.
    
    Args:
        data_dir_path: Path to the dataset directory containing class folders
        limit: Maximum number of classes to scan (None for all classes)
    
    Returns:
        dict: Dictionary mapping video file paths to class labels
    """
    data_path = Path(data_dir_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir_path}")
    
    result = {}
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    
    # Get all subdirectories (each is a class)
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {data_dir_path}")
    
    print(f"\nScanning dataset: {data_dir_path}")
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    
    dir_count = 0
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        
        # Find all video files in this class directory
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(class_dir.glob(f'*{ext}')))
        
        print(f"  Class '{class_name}': {len(video_files)} videos")
        
        for video_file in video_files:
            result[str(video_file)] = class_name
        
        dir_count += 1
        
        if limit is not None and dir_count >= limit:
            break
    
    total_videos = len(result)
    print(f"Total videos found: {total_videos}\n")
    
    if total_videos == 0:
        raise ValueError(f"No video files found in {data_dir_path}")
    
    return result


def scan_dataset_with_labels(data_dir_path, labels):
    """
    Scan dataset for videos with specific labels.
    
    Args:
        data_dir_path: Path to the dataset directory containing class folders
        labels: List of class labels to scan for
    
    Returns:
        dict: Dictionary mapping video file paths to class labels
    """
    data_path = Path(data_dir_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir_path}")
    
    result = {}
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    
    print(f"\nScanning dataset: {data_dir_path}")
    print(f"Looking for classes: {labels}")
    
    for label in labels:
        class_dir = data_path / label
        
        if not class_dir.exists() or not class_dir.is_dir():
            print(f"  Warning: Class directory '{label}' not found")
            continue
        
        # Find all video files in this class directory
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(class_dir.glob(f'*{ext}')))
        
        print(f"  Class '{label}': {len(video_files)} videos")
        
        for video_file in video_files:
            result[str(video_file)] = label
    
    total_videos = len(result)
    print(f"Total videos found: {total_videos}\n")
    
    return result


def get_class_names(data_dir_path):
    """
    Get list of class names from dataset directory.
    
    Args:
        data_dir_path: Path to the dataset directory
    
    Returns:
        list: List of class names
    """
    data_path = Path(data_dir_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir_path}")
    
    class_dirs = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    return sorted(class_dirs)


def load_dataset(data_dir_path):
    """
    Load and verify dataset exists.
    This is a no-op for custom datasets (no download needed).
    
    Args:
        data_dir_path: Path to the dataset directory
    """
    data_path = Path(data_dir_path)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir_path}\n"
            f"Please ensure your dataset is at the correct location."
        )
    
    # Verify it has subdirectories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        raise ValueError(
            f"No class directories found in {data_dir_path}\n"
            f"Expected structure: {data_dir_path}/[CLASS_NAME]/[VIDEO_FILES]"
        )


def main():
    """Test the data loader"""
    # Example usage
    data_dir_path = r"D:\projects\01\dataset\autism_data_anonymized\training_set"
    
    print("="*70)
    print("Testing ASD Data Loader")
    print("="*70)
    
    try:
        # Test load_dataset
        load_dataset(data_dir_path)
        
        # Test get_class_names
        classes = get_class_names(data_dir_path)
        print(f"Classes found: {classes}")
        
        # Test scan_dataset
        videos = scan_dataset(data_dir_path)
        print(f"\nScanned {len(videos)} total videos")
        
        # Show some examples
        print("\nSample videos:")
        for video_path, label in list(videos.items())[:5]:
            print(f"  {Path(video_path).name} -> {label}")
        
        print("\n" + "="*70)
        print("Data loader test successful!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease update data_dir_path in this script to point to your dataset.")


if __name__ == '__main__':
    main()

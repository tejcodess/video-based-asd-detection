"""
Mini Dataset Creator for Quick Testing
Creates a tiny dataset (5 videos per class) for fast prototyping
"""

import os
import shutil
from pathlib import Path
import random

# Paths
FULL_DATASET = r"D:\projects\01\dataset\autism_data_anonymized"
MINI_DATASET = r"D:\projects\01\Video-Neural-Network-ASD-screening\mini_dataset"

# Number of videos for quick test
NUM_TRAIN_PER_CLASS = 5  # 5 ASD + 5 TD = 10 training videos
NUM_TEST_PER_CLASS = 3   # 3 ASD + 3 TD = 6 testing videos

def copy_random_videos(src_folder, dst_folder, num_videos):
    """Copy random videos from src to dst"""
    os.makedirs(dst_folder, exist_ok=True)
    
    # Get all video files
    video_exts = ['.avi', '.mp4', '.mov', '.mkv']
    all_videos = []
    for ext in video_exts:
        all_videos.extend(list(Path(src_folder).glob(f'*{ext}')))
    
    if len(all_videos) == 0:
        print(f"WARNING: No videos found in {src_folder}")
        return 0
    
    # Select random videos
    num_to_copy = min(num_videos, len(all_videos))
    selected = random.sample(all_videos, num_to_copy)
    
    # Copy videos
    for video in selected:
        dst_path = Path(dst_folder) / video.name
        if not dst_path.exists():
            shutil.copy2(video, dst_path)
            print(f"  Copied: {video.name}")
    
    return num_to_copy

def main():
    print("\n" + "="*70)
    print("CREATING MINI DATASET FOR QUICK TESTING")
    print("="*70)
    print(f"Source: {FULL_DATASET}")
    print(f"Destination: {MINI_DATASET}")
    print()
    
    # Check if source exists
    if not os.path.exists(FULL_DATASET):
        print(f"ERROR: Source dataset not found at {FULL_DATASET}")
        print("Please check the path in this script.")
        return
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create training set
    print("[1/4] Creating training set - ASD class...")
    train_asd_src = os.path.join(FULL_DATASET, "training_set", "ASD")
    train_asd_dst = os.path.join(MINI_DATASET, "training_set", "ASD")
    asd_train_count = copy_random_videos(train_asd_src, train_asd_dst, NUM_TRAIN_PER_CLASS)
    
    print("\n[2/4] Creating training set - TD class...")
    train_td_src = os.path.join(FULL_DATASET, "training_set", "TD")
    train_td_dst = os.path.join(MINI_DATASET, "training_set", "TD")
    td_train_count = copy_random_videos(train_td_src, train_td_dst, NUM_TRAIN_PER_CLASS)
    
    # Create testing set
    print("\n[3/4] Creating testing set - ASD class...")
    test_asd_src = os.path.join(FULL_DATASET, "testing_set", "ASD")
    test_asd_dst = os.path.join(MINI_DATASET, "testing_set", "ASD")
    asd_test_count = copy_random_videos(test_asd_src, test_asd_dst, NUM_TEST_PER_CLASS)
    
    print("\n[4/4] Creating testing set - TD class...")
    test_td_src = os.path.join(FULL_DATASET, "testing_set", "TD")
    test_td_dst = os.path.join(MINI_DATASET, "testing_set", "TD")
    td_test_count = copy_random_videos(test_td_src, test_td_dst, NUM_TEST_PER_CLASS)
    
    # Summary
    total_train = asd_train_count + td_train_count
    total_test = asd_test_count + td_test_count
    total_all = total_train + total_test
    
    print("\n" + "="*70)
    print("MINI DATASET CREATED SUCCESSFULLY")
    print("="*70)
    print(f"Training set: {asd_train_count} ASD + {td_train_count} TD = {total_train} videos")
    print(f"Testing set:  {asd_test_count} ASD + {td_test_count} TD = {total_test} videos")
    print(f"Total: {total_all} videos")
    print(f"\nLocation: {MINI_DATASET}")
    
    # Estimate size
    if total_all > 0:
        avg_video_size_mb = 50  # Rough estimate
        total_size_mb = total_all * avg_video_size_mb
        print(f"\nEstimated disk space: ~{total_size_mb} MB")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Ensure config.py has: USE_MINI_DATASET = True")
    print("2. Run: python quick_test.py")
    print("3. Expected time: 5-15 minutes on GPU")
    print("="*70)
    print("\nThis mini dataset is perfect for quick testing before full training!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

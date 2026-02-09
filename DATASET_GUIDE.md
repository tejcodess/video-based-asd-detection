# Dataset Preparation Guide

Complete guide for organizing and preparing video datasets for ASD detection model training.

## Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Preparing Your Videos](#preparing-your-videos)
- [OpenPose Processing](#openpose-processing)
- [Dataset Organization](#dataset-organization)
- [Best Practices](#best-practices)
- [Common Issues](#common-issues)
- [Privacy & Ethics](#privacy--ethics)
- [Examples](#examples)
- [Next Steps](#next-steps)

---

## Overview

This project requires videos organized into two classes:
- **ASD**: Autism Spectrum Disorder
- **TD**: Typically Developing (controls)

Videos should be separated into training and testing sets with balanced class distribution.

### What the Model Learns

The model analyzes:
- **Visual features**: Body movements, postures, gestures
- **Temporal patterns**: Changes in behavior over time
- **Sequential information**: How behaviors evolve during video

---

## Dataset Structure

### Required Folder Structure

```
dataset/
├── training_set/
│   ├── ASD/
│   │   ├── subject1_video1.mp4
│   │   ├── subject1_video2.mp4
│   │   ├── subject2_video1.mp4
│   │   └── ...
│   └── TD/
│       ├── control1_video1.mp4
│       ├── control1_video2.mp4
│       ├── control2_video1.mp4
│       └── ...
└── testing_set/
    ├── ASD/
    │   ├── test_subject1.mp4
    │   └── ...
    └── TD/
        ├── test_control1.mp4
        └── ...
```

### Key Requirements

1. **Two main folders**: `training_set` and `testing_set`
2. **Two subfolders in each**: `ASD` and `TD`
3. **Video files**: `.mp4`, `.avi`, `.mov` formats supported
4. **No overlap**: Testing videos must not appear in training set

---

## Preparing Your Videos

### Video Format Specifications

| Property | Recommended | Minimum | Notes |
|----------|-------------|---------|-------|
| **Resolution** | 640x480 or higher | 224x224 | Will be resized to 224x224 |
| **Frame rate** | 30 fps | 15 fps | Model extracts 1 frame/sec |
| **Duration** | 10-60 seconds | 5 seconds | Longer videos = more data |
| **Format** | .mp4 (H.264) | .avi, .mov | Any OpenCV-readable format |
| **Quality** | High bitrate | Standard | Clear body visibility needed |

### Recording Guidelines

**DO:**
- ✓ Ensure subject is clearly visible
- ✓ Use consistent lighting
- ✓ Record full body or upper body
- ✓ Maintain stable camera position
- ✓ Record natural behaviors
- ✓ Use uniform background (for OpenPose processing)

**DON'T:**
- ✗ Avoid heavy occlusions
- ✗ Don't use very low resolution
- ✗ Avoid extreme lighting conditions
- ✗ Don't crop too tightly
- ✗ Avoid shaky/moving camera
- ✗ Don't include multiple subjects simultaneously

---

## OpenPose Processing

This project was designed for **OpenPose skeletal keypoint videos** showing body movements on blank backgrounds.

### Why OpenPose?

OpenPose extracts skeletal keypoints (pose estimation) which:
- Removes personally identifiable information (privacy protection)
- Focuses on body movements and postures
- Reduces background noise and distractions
- Provides consistent input format

<p align="center">
<img src="https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening/blob/main/illustrations/openpose.jpg" width="500">
</p>

### Using OpenPose

**Option 1: OpenPose Software (Recommended)**

1. Download OpenPose: [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
2. Install following their documentation
3. Process videos:

```bash
# Windows
bin\OpenPoseDemo.exe --video input.mp4 --write_video output.avi --display 0

# Linux
./build/examples/openpose/openpose.bin --video input.mp4 --write_video output.avi --display 0
```

**Option 2: Python OpenPose Wrapper**

```python
import cv2
from openpose import pyopenpose as op

# Configure OpenPose
params = {
    "model_folder": "models/",
    "video": "input.mp4"
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
```

**Option 3: Mediapipe (Alternative)**

```python
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
cap = cv2.VideoCapture("input.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Draw pose on blank background
    blank = np.zeros_like(frame)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(blank, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Save frame
    # ...
```

### Without OpenPose

If you don't have OpenPose processed videos, the model can work with:
- Regular videos of subjects
- Videos with visible body movements
- Videos focusing on upper body/torso

**⚠️ Note**: Results may vary compared to OpenPose-processed videos due to background noise and additional visual information.

---

## Dataset Organization

### Step-by-Step Organization

#### 1. Create Directory Structure

```bash
# Windows PowerShell
mkdir dataset\training_set\ASD
mkdir dataset\training_set\TD
mkdir dataset\testing_set\ASD
mkdir dataset\testing_set\TD

# Linux/macOS
mkdir -p dataset/training_set/{ASD,TD}
mkdir -p dataset/testing_set/{ASD,TD}
```

#### 2. Split Data (80/20 Rule)

- **Training**: 80% of videos (used to train model)
- **Testing**: 20% of videos (held out for evaluation)

**Important**: Ensure no subject appears in both training and testing sets!

#### 3. Copy Videos

```bash
# Windows PowerShell
Copy-Item "source\asd_videos\*.mp4" "dataset\training_set\ASD\"
Copy-Item "source\td_videos\*.mp4" "dataset\training_set\TD\"

# Linux/macOS
cp source/asd_videos/*.mp4 dataset/training_set/ASD/
cp source/td_videos/*.mp4 dataset/training_set/TD/
```

#### 4. Verify Structure

```bash
# Count videos in each folder
# Windows PowerShell
(Get-ChildItem "dataset\training_set\ASD\*.mp4").Count
(Get-ChildItem "dataset\training_set\TD\*.mp4").Count

# Linux/macOS
ls dataset/training_set/ASD/*.mp4 | wc -l
ls dataset/training_set/TD/*.mp4 | wc -l
```

---

## Best Practices

### Dataset Size Recommendations

| Dataset Size | Training Videos per Class | Expected Performance | Use Case |
|--------------|---------------------------|----------------------|----------|
| **Mini** | 10-20 | 60-70% accuracy | Testing, proof of concept |
| **Small** | 50-100 | 70-80% accuracy | Initial experiments |
| **Medium** | 100-200 | 80-90% accuracy | Research, development |
| **Large** | 200+ | 85-95% accuracy | Production, publication |

### Class Balance

Maintain balanced classes to avoid bias:

```
✓ GOOD:
  ASD: 100 videos
  TD:  95 videos
  
✗ BAD:
  ASD: 200 videos
  TD:  30 videos
```

**If unbalanced**: Use data augmentation or class weights in training.

### Subject Distribution

- **Diverse ages**: Include range of ages (if applicable)
- **Multiple videos per subject**: OK for training, but split subjects between train/test
- **Different contexts**: Various settings, activities, times

### Video Segmentation

For long videos (>1 minute), consider segmentation:

```python
# Example: Split 60-second video into 5-second clips
import cv2

cap = cv2.VideoCapture("long_video.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
segment_duration = 5  # seconds

segment_frames = fps * segment_duration
frame_count = 0
segment_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Write frame to current segment
    # ... (implement video writer logic)
    
    frame_count += 1
    if frame_count % segment_frames == 0:
        segment_num += 1
        # Start new segment file
```

**Benefits**:
- More training samples
- Better temporal granularity
- Easier to aggregate predictions

---

## Common Issues

### Issue 1: Videos Not Detected

**Symptom**: "No videos found" error

**Solutions**:
- Check folder structure matches requirements
- Verify video file extensions (.mp4, .avi)
- Ensure videos are not corrupted
- Check file permissions

### Issue 2: Codec Errors

**Symptom**: "Cannot open video codec" error

**Solutions**:
```bash
# Install additional codecs (Ubuntu)
sudo apt install ubuntu-restricted-extras

# Re-encode videos (if needed)
ffmpeg -i input.avi -vcodec libx264 -acodec aac output.mp4
```

### Issue 3: Class Imbalance Warning

**Symptom**: Model performs poorly on minority class

**Solutions**:
- Collect more videos for minority class
- Use data augmentation
- Apply class weights in training

### Issue 4: Insufficient Data

**Symptom**: Overfitting, poor generalization

**Solutions**:
- Collect more diverse videos
- Use data augmentation (flip, rotate, crop)
- Start with transfer learning (already implemented)
- Reduce model complexity

### Issue 5: Subject Overlap

**Symptom**: Unrealistically high test accuracy

**Solutions**:
- Ensure subjects in test set don't appear in training
- Use subject-wise splitting, not random video splitting
- Document subject IDs to track splits

---

## Privacy & Ethics

### Data Collection

1. **Informed Consent**
   - Obtain written consent from participants or legal guardians
   - Explain data usage and retention policies
   - Allow opt-out options

2. **Anonymization**
   - Remove personally identifiable information (PII)
   - Use subject IDs instead of names
   - OpenPose processing removes facial features
   - Avoid recording voices (if not needed)

3. **Secure Storage**
   - Encrypt sensitive data
   - Use access controls
   - Regular security audits
   - Secure deletion when no longer needed

### Ethical Considerations

**DO:**
- ✓ Follow IRB/ethics board approval if required
- ✓ Comply with HIPAA, GDPR, or local regulations
- ✓ Use data only as consented
- ✓ Document data sources and methods
- ✓ Consider potential biases in data collection

**DON'T:**
- ✗ Share sensitive data publicly
- ✗ Use data without proper consent
- ✗ Make clinical diagnoses without validation
- ✗ Deploy without clinical oversight (if medical use)

### Research Use

This model is a **research tool**, not a clinical diagnostic:
- Should not replace professional assessment
- Requires clinical validation before medical use
- Results should be interpreted by qualified professionals
- Consider cultural and demographic factors

---

## Examples

### Example 1: ADOS Clinical Videos

```
dataset/
├── training_set/
│   ├── ASD/
│   │   ├── patient_001_ados_segment1.avi
│   │   ├── patient_001_ados_segment2.avi
│   │   ├── patient_003_ados_segment1.avi
│   │   └── ... (80 more videos)
│   └── TD/
│       ├── control_002_ados_segment1.avi
│       ├── control_002_ados_segment2.avi
│       └── ... (80 more videos)
└── testing_set/
    ├── ASD/ (20 videos from different patients)
    └── TD/ (20 videos from different controls)
```

### Example 2: Naturalistic Play Videos

```
dataset/
├── training_set/
│   ├── ASD/
│   │   ├── asd_child_01_freeplay.mp4
│   │   ├── asd_child_02_structured.mp4
│   │   └── ...
│   └── TD/
│       ├── td_child_01_freeplay.mp4
│       ├── td_child_02_structured.mp4
│       └── ...
└── testing_set/
    └── ...
```

### Example 3: Using Provided Sample Datasets

The repository includes sample datasets:

```python
# In config.py, use mini_dataset for testing
DATASET_BASE = r"D:\projects\01\Video-Neural-Network-ASD-screening\mini_dataset"

# Or use dataset_20percent for development
DATASET_BASE = r"D:\projects\01\Video-Neural-Network-ASD-screening\dataset_20percent"
```

---

## Dataset Validation Script

Create a script to verify your dataset:

```python
import os
from pathlib import Path

def validate_dataset(dataset_path):
    """Validate dataset structure and contents"""
    
    issues = []
    stats = {
        'train_asd': 0, 'train_td': 0,
        'test_asd': 0, 'test_td': 0
    }
    
    # Check structure
    required_dirs = [
        'training_set/ASD', 'training_set/TD',
        'testing_set/ASD', 'testing_set/TD'
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            issues.append(f"Missing directory: {dir_path}")
    
    # Count videos
    for split in ['training_set', 'testing_set']:
        for class_name in ['ASD', 'TD']:
            path = os.path.join(dataset_path, split, class_name)
            if os.path.exists(path):
                videos = list(Path(path).glob('*.mp4')) + list(Path(path).glob('*.avi'))
                count = len(videos)
                key = f"{split.split('_')[0]}_{class_name.lower()}"
                stats[key] = count
                
                if count == 0:
                    issues.append(f"No videos in {split}/{class_name}")
    
    # Check balance
    train_balance = abs(stats['train_asd'] - stats['train_td']) / max(stats['train_asd'], stats['train_td'], 1)
    if train_balance > 0.3:
        issues.append(f"Training set imbalanced: ASD={stats['train_asd']}, TD={stats['train_td']}")
    
    # Print report
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Training set:")
    print(f"  ASD: {stats['train_asd']} videos")
    print(f"  TD:  {stats['train_td']} videos")
    print(f"Testing set:")
    print(f"  ASD: {stats['test_asd']} videos")
    print(f"  TD:  {stats['test_td']} videos")
    print(f"")
    
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Dataset structure is valid!")
    print("=" * 60)
    
    return len(issues) == 0

# Usage
if __name__ == "__main__":
    validate_dataset(r"D:\path\to\your\dataset")
```

Save this as `validate_dataset.py` and run before training.

---

## Next Steps

After organizing your dataset:

1. **Update config.py**:
```python
DATASET_BASE = r"D:\path\to\your\dataset"
```

2. **Validate dataset**:
```bash
python validate_dataset.py
```

3. **Start training**:
```bash
python train_asd_model.py
```

4. **Monitor training**:
- Check console output for progress
- Review training plots in `reports/`
- Examine model checkpoints in `models/`

---

## Additional Resources

- **OpenPose Documentation**: [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- **Video Processing with OpenCV**: [https://docs.opencv.org/master/d6/d00/tutorial_py_root.html](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- **Data Augmentation**: Consider imgaug or albumentations libraries
- **HIPAA Compliance**: [https://www.hhs.gov/hipaa/](https://www.hhs.gov/hipaa/)
- **GDPR Guidelines**: [https://gdpr.eu/](https://gdpr.eu/)

---

## Getting Help

For dataset-related questions:

1. Check this guide thoroughly
2. Review [README.md](README.md) for general information
3. Search [GitHub Issues](https://github.com/nshreyasvi/Video-Neural-Network-ASD-screening/issues)
4. Create a new issue with dataset structure details

---

**Dataset preparation complete!** You're ready to train your ASD detection model.

*For installation help, see [INSTALLATION.md](INSTALLATION.md)*

import cv2
import os
import numpy as np
import gc
from pathlib import Path
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
import config # Import our updated config

# Try to import MediaPipe for normalization
try:
    import mediapipe as mp
except ImportError:
    mp = None

MAX_NB_CLASSES = 2

class PoseNormalizer:
    """Handles conversion of raw video to OpenPose-style skeletal video."""
    def __init__(self):
        if mp is None:
            raise ImportError("MediaPipe is required for normalization. Run: pip install mediapipe")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def is_normalized(self, video_path):
        """
        Checks if the video is already normalized.
        Heuristic: Normalized videos are stored in the NORMALIZED_VIDEOS_PATH.
        """
        return config.NORMALIZED_VIDEOS_PATH in str(video_path)

    def process_video(self, input_path, output_path):
        """Converts raw video to skeletons on a black background."""
        if os.path.exists(output_path):
            return output_path

        print(f"    [Normalize] Converting to pose: {Path(input_path).name}")
        cap = cv2.VideoCapture(input_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            # Create black canvas
            black_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Process pose
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    black_image, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                )
            
            out.write(black_image)
            
        cap.release()
        out.release()
        return output_path

def extract_vgg16_features(model, video_input_file_path, feature_output_file_path):
    if os.path.exists(feature_output_file_path):
        print(f'  Loading cached features: {Path(feature_output_file_path).name}')
        return np.load(feature_output_file_path)

    print(f'  Extracting frames from video: {Path(video_input_file_path).name}')
    vidcap = cv2.VideoCapture(video_input_file_path)
    
    if not vidcap.isOpened():
        raise ValueError(f"Cannot open video file: {video_input_file_path}")
    
    features = []
    success = True
    count = 0
    while success:
        # Extract at intervals defined in config (default 1s)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000 / config.FRAME_RATE))  
        success, image = vidcap.read()
        if success:
            img = cv2.resize(image, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            input_data = img_to_array(img)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = preprocess_input(input_data)
            feature = model.predict(input_data, verbose=0).ravel()
            features.append(feature)
            count = count + 1
        gc.collect()
    vidcap.release()
    
    unscaled_features = np.array(features)
    np.save(feature_output_file_path, unscaled_features)
    print(f'    Extracted {count} frames, shape: {unscaled_features.shape}')
    return unscaled_features

def scan_and_extract_vgg16_features(data_dir_path, output_dir_path, model=None, data_set_name=None):
    input_data_dir_path = Path(data_dir_path)
    base_dir = input_data_dir_path.parent
    output_feature_data_dir_path = base_dir / output_dir_path
    
    # Initialize Normalizer if enabled
    normalizer = PoseNormalizer() if config.ENABLE_POSE_NORMALIZATION else None

    if model is None:
        print("Creating VGG16 model...")
        model = VGG16(include_top=config.VGG16_INCLUDE_TOP, weights='imagenet')
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    os.makedirs(output_feature_data_dir_path, exist_ok=True)
    
    x_samples, y_samples = [], []
    class_dirs = [d for d in input_data_dir_path.iterdir() if d.is_dir()]
    
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        output_class_dir = output_feature_data_dir_path / class_name
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Setup folder for normalized videos
        norm_class_dir = Path(config.NORMALIZED_VIDEOS_PATH) / class_name
        os.makedirs(norm_class_dir, exist_ok=True)
        
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(class_dir.glob(f'*{ext}')))
        
        print(f"\nProcessing class '{class_name}': {len(video_files)} videos")
        
        for video_file in sorted(video_files):
            video_file_path = str(video_file)
            
            # 1. Gatekeeper: Normalize if needed
            if normalizer and not normalizer.is_normalized(video_file_path):
                norm_video_path = norm_class_dir / f"{video_file.stem}_norm.mp4"
                video_to_extract = normalizer.process_video(video_file_path, str(norm_video_path))
            else:
                video_to_extract = video_file_path

            # 2. Extract Features from the (potentially new) video
            output_feature_file_path = str(output_class_dir / f"{video_file.stem}.npy")
            try:
                x = extract_vgg16_features(model, video_to_extract, output_feature_file_path)
                y_samples.append(class_name)
                x_samples.append(x)
            except Exception as e:
                print(f"  ✗ Error: {e}")
            
            gc.collect()
            
    return x_samples, y_samples
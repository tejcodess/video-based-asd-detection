import cv2
import mediapipe as mp
import numpy as np
import os

class VideoPreprocessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def is_normalized(self, video_path):
        # Implementation 1: Check filename suffix
        if "_normalized" in video_path:
            return True
        
        # Implementation 2: Visual check (optional/heuristic)
        # Check if the first frame is mostly black (typical for normalized pose videos)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return np.mean(frame) < 30 # Threshold for "mostly black"
        return False

    def normalize_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Create a black background
            black_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Extract pose and draw on black background
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    black_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            out.write(black_image)
            
        cap.release()
        out.release()
        return output_path
import cv2
import numpy as np
import os

# Create directory structure
os.makedirs('very_large_data/autism_data/class_0', exist_ok=True)
os.makedirs('very_large_data/autism_data/class_1', exist_ok=True)

print("Creating sample video files...")

# Create sample videos for each class
for cls in ['class_0', 'class_1']:
    for i in range(2):  # 2 videos per class
        filename = f'very_large_data/autism_data/{cls}/sample_{i}.mp4'
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 10.0, (224, 224))
        
        # Write 30 frames (3 seconds at 10 fps)
        for frame_num in range(30):
            # Create a random frame
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        print(f"Created: {filename}")

print("\nSample video files created successfully!")
print("Structure:")
print("  very_large_data/autism_data/")
print("    ├── class_0/ (2 videos)")
print("    └── class_1/ (2 videos)")

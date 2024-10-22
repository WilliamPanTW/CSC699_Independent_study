from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load the YOLOv8 model
model = YOLO(r'D:\CSC699_Independent_study\application\yolov8\yolov8_custom.pt')

# Path to the video file
video_path = r"D:\CSC699_Independent_study\application\GUI\MVI_7029.MP4"

# Path to save the final frame
save_dir = r"D:\CSC699_Independent_study\application\GUI"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Create directory if it doesn't exist

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize variables
frame_number = 0
best_frame = None
best_confidence = 0  # Track the highest confidence
best_box = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO model on the current frame
    results = model.predict(source=frame, conf=0.5)
    current_boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
    current_confidences = results[0].boxes.conf.cpu().numpy()  # Get confidence scores
    
    # Find the highest confidence detection in this frame
    if len(current_confidences) > 0:
        max_conf_idx = np.argmax(current_confidences)
        max_conf = current_confidences[max_conf_idx]
        
        # Update the best frame if this frame has the highest confidence detection so far
        if max_conf > best_confidence:
            best_confidence = max_conf
            best_frame = frame
            best_box = current_boxes[max_conf_idx]  # Get the bounding box for the highest confidence
    
    frame_number += 1

# Save the best frame with the highest confidence score
if best_frame is not None:
    save_path = os.path.join(save_dir, f"test.jpg")
    cv2.imwrite(save_path, best_frame)
    print(f"Saved the best frame with the highest confidence score at: {save_path}")
    print(f"Bounding Box: {best_box}, Confidence: {best_confidence}")
else:
    print("No suitable frame found.")

# Release video capture
cap.release()
cv2.destroyAllWindows()

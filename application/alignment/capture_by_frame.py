from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load the YOLOv8 model
model = YOLO(r'D:\CSC699_Independent_study\application\yolov8\yolov8_custom.pt')

# Path to the video file
video_path = r"D:\CSC699_Independent_study\application\yolov8\SLATE.mp4"

# Path to save the final frame
save_dir = r"D:\CSC699_Independent_study\application\yolov8\results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Create directory if it doesn't exist

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_number = 0
prev_boxes = []
stability_count = {}
best_frame = None
best_box = None
max_stability = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO model on the current frame
    results = model.predict(source=frame, conf=0.5)
    current_boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
    
    # Compare the current boxes with previous ones to check for stability
    for box in current_boxes:
        x1, y1, x2, y2 = box[:4]
        found_match = False
        
        # Compare with previous boxes
        for prev_box in prev_boxes:
            px1, py1, px2, py2 = prev_box[:4]
            # Calculate the distance between boxes to check if they are similar
            if np.linalg.norm([x1 - px1, y1 - py1, x2 - px2, y2 - py2]) < 10:  # Threshold of 10 pixels
                found_match = True
                box_tuple = tuple(prev_box)
                stability_count[box_tuple] = stability_count.get(box_tuple, 0) + 1
                
                # Update the most stable box
                if stability_count[box_tuple] > max_stability:
                    max_stability = stability_count[box_tuple]
                    best_frame = frame
                    best_box = box_tuple
                break
        
        if not found_match:
            # Add new box if no match found
            box_tuple = tuple(box[:4])
            stability_count[box_tuple] = 1
    
    prev_boxes = current_boxes  # Update previous boxes to the current ones
    frame_number += 1

# Save the best frame with the most stable box
if best_frame is not None:
    save_path = os.path.join(save_dir, f"best_frame.jpg")
    cv2.imwrite(save_path, best_frame)
    print(f"Saved the best frame with the most stable box at: {save_path}")
else:
    print("No stable box found.")

# Release video capture
cap.release()
cv2.destroyAllWindows()

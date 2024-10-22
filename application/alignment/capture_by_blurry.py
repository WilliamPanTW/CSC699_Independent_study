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

# Parameters for detecting blur
def detect_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Variance of Laplacian
    return laplacian_var

# Initialize variables
frame_number = 0
prev_boxes = []
stability_count = {}
best_frame = None
best_box = None
max_score = 0

blur_threshold = 100  # Threshold to determine if a frame is sharp enough
stability_threshold = 10  # Threshold for bounding box position change (pixels)
confidence_weight = 0.5  # Weight of detection confidence
stability_weight = 0.3  # Weight of bounding box stability
sharpness_weight = 0.2  # Weight of frame sharpness

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO model on the current frame
    results = model.predict(source=frame, conf=0.5)
    current_boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
    current_confidences = results[0].boxes.conf.cpu().numpy()  # Get confidence scores
    
    # Detect frame blur
    blur_score = detect_blur(frame)
    
    # Compare the current boxes with previous ones to check for stability
    for i, box in enumerate(current_boxes):
        x1, y1, x2, y2 = box[:4]
        found_match = False
        
        # Compare with previous boxes
        for prev_box in prev_boxes:
            px1, py1, px2, py2 = prev_box[:4]
            # Calculate the distance between boxes to check if they are similar
            if np.linalg.norm([x1 - px1, y1 - py1, x2 - px2, y2 - py2]) < stability_threshold:
                found_match = True
                box_tuple = tuple(prev_box)
                
                # Update stability count for the matched box
                stability_count[box_tuple] = stability_count.get(box_tuple, 0) + 1
                
                # Calculate the weighted score
                stability_score = stability_count[box_tuple]
                confidence_score = current_confidences[i]
                total_score = (confidence_weight * confidence_score +
                               stability_weight * stability_score +
                               sharpness_weight * blur_score)
                
                # Update the best frame and box if this score is the highest
                if total_score > max_score and blur_score > blur_threshold:
                    max_score = total_score
                    best_frame = frame
                    best_box = box_tuple
                break
        
        if not found_match:
            # Add new box if no match found
            box_tuple = tuple(box[:4])
            stability_count[box_tuple] = 1
    
    prev_boxes = current_boxes  # Update previous boxes to the current ones
    frame_number += 1

# Save the best frame with the highest weighted score
if best_frame is not None:
    save_path = os.path.join(save_dir, f"best_frame.jpg")
    cv2.imwrite(save_path, best_frame)
    print(f"Saved the best frame with the highest weighted score at: {save_path}")
else:
    print("No suitable frame found.")

# Release video capture
cap.release()
cv2.destroyAllWindows()

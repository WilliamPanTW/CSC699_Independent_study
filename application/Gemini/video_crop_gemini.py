from ultralytics import YOLO
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure the API key for Generative AI
genai.configure(api_key=os.environ["API_KEY"])

# Initialize the YOLOv8 model
model = YOLO(r'D:\CSC699_Independent_study\application\yolov8\yolov8_custom.pt')

# Path to the video file
video_path = r"D:\CSC699_Independent_study\application\GUI\MVI_7030.MP4"

# Path to save the final frame
save_dir = r"D:\CSC699_Independent_study\application\Gemini"
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
            best_frame = frame.copy()
            best_box = current_boxes[max_conf_idx]  # Get the bounding box for the highest confidence
    
    frame_number += 1

# Save the best frame with the highest confidence score and crop the detected region
if best_frame is not None:
    # Convert bounding box coordinates to integers
    x_min, y_min, x_max, y_max = map(int, best_box)
    
    # Crop the detected region using bounding box coordinates
    cropped_img = best_frame[y_min:y_max, x_min:x_max]
    
    # Save the cropped image
    cropped_save_path = os.path.join(save_dir, f"cropped_best_frame.jpg")
    cv2.imwrite(cropped_save_path, cropped_img)
    
    # Save the full best frame (optional)
    full_frame_save_path = os.path.join(save_dir, f"best_frame.jpg")
    cv2.imwrite(full_frame_save_path, best_frame)
    
    print(f"Saved the best frame with the highest confidence score at: {full_frame_save_path}")
    print(f"Saved cropped detected region at: {cropped_save_path}")
    print(f"Bounding Box: {best_box}, Confidence: {best_confidence}")
    
    # Upload cropped image to Generative AI for content generation
    myfile = genai.upload_file(cropped_save_path)
    print(f"Uploaded file: {cropped_save_path}")
    
    # Initialize Generative AI model
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Generate content based on the cropped image
    result = gemini_model.generate_content(
        [myfile, "what is the scene, take, and roll number?"]
    )
    
    # Print the result
    print(f"Generated Content: {result.text}")
else:
    print("No suitable frame found.")

# Release video capture
cap.release()

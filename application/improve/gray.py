from ultralytics import YOLO
import cv2
import os
import numpy as np
import easyocr
from matplotlib import pyplot as plt

# Load the YOLOv8 model
model = YOLO(r'D:\CSC699_Independent_study\application\yolov8\yolov8_custom.pt')

# Path to the video file
video_path = r"D:\CSC699_Independent_study\application\GUI\MVI_7029.MP4"

# Path to save the final frame
save_dir = r"D:\CSC699_Independent_study\application\GUI"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize variables
frame_number = 0
best_frame = None
best_confidence = 0
best_box = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO model on the current frame
    results = model.predict(source=frame, conf=0.5)
    current_boxes = results[0].boxes.xyxy.cpu().numpy()
    current_confidences = results[0].boxes.conf.cpu().numpy()
    
    # Find the highest confidence detection in this frame
    if len(current_confidences) > 0:
        max_conf_idx = np.argmax(current_confidences)
        max_conf = current_confidences[max_conf_idx]
        
        # Update the best frame if this frame has the highest confidence detection so far
        if max_conf > best_confidence:
            best_confidence = max_conf
            best_frame = frame.copy()
            best_box = current_boxes[max_conf_idx]
    
    frame_number += 1

# Save and process the best frame if found
if best_frame is not None:
    x_min, y_min, x_max, y_max = map(int, best_box)
    cropped_img = best_frame[y_min:y_max, x_min:x_max]
    
    # Increase contrast and apply thresholding for OCR
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 30    # Brightness control (0-100)
    enhanced_img = cv2.convertScaleAbs(cropped_img, alpha=alpha, beta=beta)

    # Convert to grayscale
    gray_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresholded_img = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Save the processed image
    processed_save_path = os.path.join(save_dir, "cropped_best_frame_processed.jpg")
    cv2.imwrite(processed_save_path, thresholded_img)
    print(f"Saved processed cropped image at: {processed_save_path}")

    # Optional: save the best frame for reference
    full_frame_save_path = os.path.join(save_dir, "best_frame.jpg")
    cv2.imwrite(full_frame_save_path, best_frame)

else:
    print("No suitable frame found.")

# Release video capture
cap.release()

# OCR on the processed image
if os.path.exists(processed_save_path):
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(processed_save_path)

    # Load processed image for display
    img = cv2.imread(processed_save_path)

    font = cv2.FONT_HERSHEY_SIMPLEX
    spacer = 100

    scene_position = None
    roll_position = None
    take_position = None
    number_positions = []

    for detection in result:
        box, text, _ = detection
        center_x = (box[0][0] + box[2][0]) / 2
        center_y = (box[0][1] + box[2][1]) / 2
        center = np.array([center_x, center_y])

        if 'SCENE' in text.upper():
            scene_position = center
        elif 'ROLL' in text.upper():
            roll_position = center
        elif 'TAKE' in text.upper():
            take_position = center
        elif text.isdigit():
            number_positions.append((center, text))

        top_left = tuple(map(int, box[0]))
        bottom_right = tuple(map(int, box[2]))
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
        img = cv2.putText(img, text, (20, spacer), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        spacer += 15

    scene_value = None
    roll_value = None

    if scene_position is not None:
        scene_value = min(number_positions, key=lambda x: np.linalg.norm(scene_position - x[0]))[1]
    if roll_position is not None:
        roll_value = min(number_positions, key=lambda x: np.linalg.norm(roll_position - x[0]))[1]

    print(f"Scene: {scene_value}")
    print(f"Roll: {roll_value}")

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("Processed image not found for OCR.")

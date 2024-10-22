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
    save_path = os.path.join(save_dir, f"best_frame.jpg")
    cv2.imwrite(save_path, best_frame)
    print(f"Saved the best frame with the highest confidence score at: {save_path}")
    print(f"Bounding Box: {best_box}, Confidence: {best_confidence}")
else:
    print("No suitable frame found.")

# Release video capture
cap.release()

# Now, process the saved best frame with EasyOCR
# Function to calculate the center of a bounding box
def calculate_center(box):
    # Get top-left and bottom-right points of the box
    top_left, bottom_right = box[0], box[2]
    # Calculate center point
    center_x = (top_left[0] + bottom_right[0]) / 2
    center_y = (top_left[1] + bottom_right[1]) / 2
    return np.array([center_x, center_y])

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Initialize the EasyOCR reader for English
reader = easyocr.Reader(['en'], gpu=True)

# Read the text from the best frame (saved image)
image_path = save_path
result = reader.readtext(image_path)

# Print the result for debugging
print(result)

# Load the image using OpenCV
img = cv2.imread(image_path)

# Define the font for text overlay
font = cv2.FONT_HERSHEY_SIMPLEX
spacer = 100  # Spacer for text positioning

# Variables to store positions of Scene, Take, Roll, and detected numbers
scene_position = None
roll_position = None
take_position = None
number_positions = []

# Iterate through the detections
for detection in result:
    box, text, _ = detection
    # Calculate the center of the bounding box
    center = calculate_center(box)

    # Check if the text is 'SCENE', 'ROLL', or 'TAKE' and store the positions
    if 'SCENE' in text.upper():
        scene_position = center
    elif 'ROLL' in text.upper():
        roll_position = center
    elif 'TAKE' in text.upper():
        take_position = center
    # Check if the text looks like a number
    elif text.isdigit():
        number_positions.append((center, text))

    # Draw the rectangle around the detected text
    top_left = tuple(map(int, box[0]))  # Convert to integers
    bottom_right = tuple(map(int, box[2]))  # Convert to integers
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
    img = cv2.putText(img, text, (20, spacer), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    spacer += 15  # Increment spacer for the next text line

# Now, find the nearest number to 'Scene' and 'Roll'
scene_value = None
roll_value = None

# Check if scene_position is not None
if scene_position is not None:
    # Find the closest number to 'Scene'
    scene_value = min(number_positions, key=lambda x: calculate_distance(scene_position, x[0]))[1]

# Check if roll_position is not None
if roll_position is not None:
    # Find the closest number to 'Roll'
    roll_value = min(number_positions, key=lambda x: calculate_distance(roll_position, x[0]))[1]

# Print the detected Scene and Roll values
print(f"Scene: {scene_value}")
print(f"Roll: {roll_value}")

# Display the image with detected text and bounding boxes using matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()

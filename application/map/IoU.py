import cv2
import easyocr
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = r"D:\CSC699_Independent_study\application\map\test.jpg"
img = cv2.imread(image_path)

# Initialize the EasyOCR reader for English
reader = easyocr.Reader(['en'], gpu=True)

# Read the text from the image
result = reader.readtext(image_path)

# Print the result for debugging
print(result)

# Function to calculate the center of a bounding box
def calculate_center(box):
    top_left, bottom_right = box[0], box[2]
    center_x = (top_left[0] + bottom_right[0]) / 2
    center_y = (top_left[1] + bottom_right[1]) / 2
    return np.array([center_x, center_y])

# Function to calculate the IoU between two bounding boxes
def calculate_iou(boxA, boxB):
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[2][0], boxB[2][0])
    yB = min(boxA[2][1], boxB[2][1])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2][0] - boxA[0][0]) * (boxA[2][1] - boxA[0][1])
    boxBArea = (boxB[2][0] - boxB[0][0]) * (boxB[2][1] - boxB[0][1])

    iou = interArea / float(boxAArea + boxBArea - interArea) if boxAArea + boxBArea - interArea > 0 else 0
    return iou

# Define the font for text overlay
font = cv2.FONT_HERSHEY_SIMPLEX
spacer = 100  # Spacer for text positioning

# Variables to store positions and bounding boxes for Scene, Take, Roll, and detected numbers
scene_position = None
roll_position = None
take_position = None
scene_box = None
roll_box = None
take_box = None
number_positions = []

# Iterate through the detections
for detection in result:
    box, text, _ = detection
    center = calculate_center(box)

    # Check if the text is 'SCENE', 'ROLL', or 'TAKE' and store the positions and bounding boxes
    if 'SCENE' in text.upper():
        scene_position = center
        scene_box = box
    elif 'ROLL' in text.upper():
        roll_position = center
        roll_box = box
    elif 'TAKE' in text.upper():
        take_position = center
        take_box = box
    # Check if the text looks like a number
    elif text.isdigit():
        number_positions.append((center, text, box))

    # Draw rectangle and text overlay on the detected image
    top_left = tuple(map(int, box[0]))  # Convert to integers
    bottom_right = tuple(map(int, box[2]))  # Convert to integers
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
    img = cv2.putText(img, text, (20, spacer), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    spacer += 15

# Helper function to find the closest number for a given keyword bounding box
def find_closest_number(keyword_box, numbers, iou_threshold=0.1):
    max_iou = 0
    closest_number = None
    closest_distance = float('inf')  # To find the nearest by distance if IoU fails

    for center, number, box in numbers:
        iou = calculate_iou(keyword_box, box)
        if iou > iou_threshold and iou > max_iou:
            max_iou = iou
            closest_number = number

    # If no match based on IoU, fallback to the closest by Euclidean distance
    if closest_number is None:
        keyword_center = calculate_center(keyword_box)
        for center, number, box in numbers:
            distance = np.linalg.norm(keyword_center - center)
            if distance < closest_distance:
                closest_distance = distance
                closest_number = number

    return closest_number

# Match Scene, Roll, and Take with the closest number based on IoU, with fallback to distance
scene_number = find_closest_number(scene_box, number_positions) if scene_box else None
roll_number = find_closest_number(roll_box, number_positions) if roll_box else None
take_number = find_closest_number(take_box, number_positions) if take_box else None

# Print out the results
print(f"Scene: {scene_number}")
print(f"Roll: {roll_number}")
print(f"Take: {take_number}")

# Display the image with detected text and bounding boxes
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

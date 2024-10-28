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

# Function to calculate the center of a bounding box
def calculate_center(box):
    top_left, bottom_right = box[0], box[2]
    center_x = (top_left[0] + bottom_right[0]) / 2
    center_y = (top_left[1] + bottom_right[1]) / 2
    return np.array([center_x, center_y])

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
    center = calculate_center(box)

    # Print out the coordinates and detected text
    print(f"Detected: {text}, Coordinates: {box}")

    # Check if the text is 'SCENE', 'ROLL', or 'TAKE' and store the positions
    if 'SCENE' in text.upper():
        scene_position = center
    elif 'ROLL' in text.upper():
        roll_position = center
    elif 'TAKE' in text.upper():
        take_position = center
    # Check if the text looks like a number
    elif text.isdigit():
        number_positions.append((center, text, box))

    # Draw rectangle and text overlay on the detected image
    top_left = tuple(map(int, box[0]))  # Convert to integers
    bottom_right = tuple(map(int, box[2]))  # Convert to integers
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
    img = cv2.putText(img, text, (top_left[0], top_left[1] - 10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    spacer += 15

# Helper function to find the closest number directly below the keyword
def find_number_below(keyword_center, numbers, vertical_offset=50):
    closest_number = None

    for center, number, box in numbers:
        # Check if the number is directly below the keyword within the vertical offset
        if center[0] == keyword_center[0] and (center[1] - keyword_center[1]) < vertical_offset:
            closest_number = number
            break  # Since we want the first number below, we can break early

    return closest_number

# Match Scene, Roll, and Take with the closest number below
scene_number = find_number_below(scene_position, number_positions) if scene_position is not None else None
roll_number = find_number_below(roll_position, number_positions) if roll_position is not None else None
take_number = find_number_below(take_position, number_positions) if take_position is not None else None

# Print out the results
print(f"Scene: {scene_number}")
print(f"Roll: {roll_number}")
print(f"Take: {take_number}")

# Display the image with detected text and bounding boxes
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

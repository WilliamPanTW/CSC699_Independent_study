import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Specify the image path
image_path = '12.jpg'

# Initialize the EasyOCR reader for English
reader = easyocr.Reader(['en'], gpu=True)

# Read the text from the image
result = reader.readtext(image_path)

# Print the result for debugging
print(result)

# Load the image using OpenCV
img = cv2.imread(image_path)

# Define the font for text overlay
font = cv2.FONT_HERSHEY_SIMPLEX
spacer = 100  # Spacer for text positioning

# Iterate through the detections
for detection in result:
    # Extract the top-left and bottom-right points of the bounding box
    top_left = tuple(map(int, detection[0][0]))  # Convert to integers
    bottom_right = tuple(map(int, detection[0][2]))  # Convert to integers
    
    text = detection[1]  # Extract the detected text

    # Draw the rectangle around the detected text
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)

    # Put the detected text on the image
    img = cv2.putText(img, text, (20, spacer), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    spacer += 15  # Increment spacer for the next text line

# Display the image using matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()
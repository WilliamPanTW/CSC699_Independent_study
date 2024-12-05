from ultralytics import YOLO
import cv2
import os
import numpy as np
import json
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# Load environment variables from .env file
load_dotenv()

# Configure the API key for Generative AI
genai.configure(api_key=os.environ["API_KEY"])

# Initialize the YOLOv8 model
model = YOLO(r'D:\CSC699_Independent_study\application\yolov8\yolov8_custom.pt')

# Initialize tkinter window
window = tk.Tk()
window.title("Frame Extraction and Generative AI")

# Path to save the final frame
save_dir = r"D:\CSC699_Independent_study\application\gemini"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Create directory if it doesn't exist

# Create label to display video frame
frame_label = tk.Label(window)
frame_label.pack()

# Create labels to display results
status_label = tk.Label(window, text="Status: Waiting for video file...", font=("Helvetica", 12))
status_label.pack()

best_frame = None
best_confidence = 0  # Track the highest confidence
best_box = None

# Function to handle file selection
def select_video_file():
    video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.MP4"), ("All files", "*.*")])
    if video_path:
        status_label.config(text="Status: Processing video...")
        process_video(video_path)

# Function to process the video
def process_video(video_path):
    global best_frame, best_confidence, best_box

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
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

    cap.release()
    
    # If we found a suitable frame, display it
    if best_frame is not None:
        display_best_frame()
        save_and_process_frame()
    else:
        status_label.config(text="Status: No suitable frame found.")

# Function to display the best frame on the GUI
def display_best_frame():
    # Convert OpenCV frame to PIL Image
    img = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Update the frame label
    frame_label.config(image=img_tk)
    frame_label.image = img_tk

# Function to save and process the best frame
def save_and_process_frame():
    global best_frame, best_box, best_confidence

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
    
    # Update status
    status_label.config(text=f"Status: Saved best frame and cropped image.")

    # Upload cropped image to Generative AI for content generation
    myfile = genai.upload_file(cropped_save_path)
    
    # Initialize Generative AI model
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Ask the model to return the attributes in a structured JSON format
    result = gemini_model.generate_content(
        [myfile, "Extract the roll, scene, and take numbers from this image and return them as a JSON object with keys 'roll', 'scene', and 'take'."]
    )
    
    # Handle Markdown-style JSON response
    raw_output = result.text
    if raw_output.startswith("```json") and raw_output.endswith("```"):
        json_text = raw_output[7:-3].strip()  # Strip "```json" and "```"
    else:
        json_text = raw_output.strip()

    # Parse the cleaned JSON text
    try:
        attributes = json.loads(json_text)  # Parse the JSON
    except json.JSONDecodeError as e:
        print(f"Error: The model response is not in valid JSON format. Details: {e}")
        attributes = {}

    # Save the attributes to a JSON file
    if attributes:
        output_file = os.path.join(save_dir, "extracted_attributes.json")
        with open(output_file, "w") as f:
            json.dump(attributes, f, indent=4)
        print(f"Extracted attributes saved to {output_file}:")
        print(attributes)
    
# Create buttons
load_button = tk.Button(window, text="Load Video", command=select_video_file)
load_button.pack(pady=10)

# Run the GUI event loop
window.mainloop()

import os
import cv2
import json
import numpy as np
from tkinter import filedialog, messagebox
import google.generativeai as genai
from dotenv import load_dotenv
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk

# Load environment variables from .env file
load_dotenv()

# Configure the API key for Generative AI
genai.configure(api_key=os.environ["API_KEY"])

# Initialize the YOLOv8 model
model = YOLO(r'D:\CSC699_Independent_study\application\yolov8\yolov8_custom.pt')

# Initialize tkinter window
window = tk.Tk()
window.title("Frame Extraction and Generative AI")

# Default path for saving results
save_dir = r"D:\CSC699_Independent_study\application\GUI"

# Create label to display video frame (optional, we won't update it automatically)
frame_label = tk.Label(window)
frame_label.pack()

# Create labels to display status and extracted attributes
status_label = tk.Label(window, text="Status: Waiting for video file...", font=("Helvetica", 12))
status_label.pack()

roll_label = tk.Label(window, text="Roll: N/A", font=("Helvetica", 10))
roll_label.pack()

scene_label = tk.Label(window, text="Scene: N/A", font=("Helvetica", 10))
scene_label.pack()

take_label = tk.Label(window, text="Take: N/A", font=("Helvetica", 10))
take_label.pack()

best_frame = None
best_confidence = 0  # Track the highest confidence
best_box = None
extracted_attributes = {}

# Function to handle file selection for video
def select_video_file():
    global save_dir, video_path
    video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.MP4"), ("All files", "*.*")])
    if video_path:
        # Use the video's directory as the default save directory
        save_dir = os.path.dirname(video_path)
        status_label.config(text="Status: Video selected. Click 'Process' to start.")
        process_button.config(state="normal")  # Enable Process button

# Function to process the video
def process_video():
    global best_frame, best_confidence, best_box, extracted_attributes

    if not save_dir:
        messagebox.showerror("Error", "Please select a video first.")
        return

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
    
    # If we found a suitable frame, process and extract attributes
    if best_frame is not None:
        save_and_process_frame()
    else:
        status_label.config(text="Status: No suitable frame found.")
        roll_label.config(text="Roll: N/A")
        scene_label.config(text="Scene: N/A")
        take_label.config(text="Take: N/A")

# Function to save and process the best frame
def save_and_process_frame():
    global best_frame, best_box, extracted_attributes

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
        extracted_attributes = json.loads(json_text)  # Parse the JSON
    except json.JSONDecodeError as e:
        print(f"Error: The model response is not in valid JSON format. Details: {e}")
        extracted_attributes = {}

    # Display extracted attributes on GUI
    if extracted_attributes:
        roll_label.config(text=f"Roll: {extracted_attributes.get('roll', 'N/A')}")
        scene_label.config(text=f"Scene: {extracted_attributes.get('scene', 'N/A')}")
        take_label.config(text=f"Take: {extracted_attributes.get('take', 'N/A')}")
    
    # Save the attributes to a JSON file
    if extracted_attributes:
        output_file = os.path.join(save_dir, "extracted_attributes.json")
        with open(output_file, "w") as f:
            json.dump(extracted_attributes, f, indent=4)
        print(f"Extracted attributes saved to {output_file}:")
        print(extracted_attributes)

    # Rename file after processing is complete
    rename_file()

# Function to rename the file
def rename_file():
    global video_path, save_dir

    # Get the new filename from the input field
    new_filename = filename_entry.get().strip()

    # Ensure a filename is entered
    if not new_filename:
        messagebox.showerror("Error", "Please enter a valid filename.")
        return

    # Check if the video file path exists
    if not os.path.exists(video_path):
        messagebox.showerror("Error", "No video file selected.")
        return

    # Construct the new file path
    file_extension = os.path.splitext(video_path)[1]  # Get the extension of the original file
    new_file_path = os.path.join(save_dir, f"{new_filename}{file_extension}")
    
    # Rename the file
    try:
        os.rename(video_path, new_file_path)
        video_path = new_file_path  # Update the video_path to the new filename
        messagebox.showinfo("Success", f"File renamed to {new_filename}{file_extension}")
        status_label.config(text=f"Status: File renamed to {new_filename}{file_extension}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to rename file. Error: {e}")

# Add a label and input field for user to enter the new filename
filename_label = tk.Label(window, text="Enter new filename (without extension):", font=("Helvetica", 10))
filename_label.pack(pady=5)

filename_entry = tk.Entry(window, font=("Helvetica", 10))
filename_entry.pack(pady=5)

# Create buttons
load_button = tk.Button(window, text="Load Video", command=select_video_file)
load_button.pack(pady=10)

process_button = tk.Button(window, text="Process Video", command=process_video, state="disabled")
process_button.pack(pady=10)

# Run the GUI event loop
window.mainloop()

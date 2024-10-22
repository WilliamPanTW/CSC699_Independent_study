import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import easyocr
from matplotlib import pyplot as plt

# Load the YOLOv8 model
model = YOLO(r'D:\CSC699_Independent_study\application\yolov8\yolov8_custom.pt')

# Path to save the final frame
save_dir = r"D:\CSC699_Independent_study\application\GUI"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Create directory if it doesn't exist

# Function to calculate the center of a bounding box
def calculate_center(box):
    top_left, bottom_right = box[0], box[2]
    center_x = (top_left[0] + bottom_right[0]) / 2
    center_y = (top_left[1] + bottom_right[1]) / 2
    return np.array([center_x, center_y])

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Function to open file dialog and allow selection of .mp4
def open_file():
    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4")]
    )
    if file_path:
        file_label.config(text=f"Selected File: {file_path}")
        global selected_file
        selected_file = file_path
        play_video(file_path)

# Function to rename the selected file
def rename_file():
    new_name = entry.get()
    if not selected_file:
        messagebox.showerror("Error", "No file selected.")
        return
    if not new_name:
        messagebox.showerror("Error", "No new name provided.")
        return

    file_extension = os.path.splitext(selected_file)[1]
    new_file_path = os.path.join(os.path.dirname(selected_file), new_name + file_extension)

    try:
        os.rename(selected_file, new_file_path)
        messagebox.showinfo("Success", f"File renamed to: {new_file_path}")
        file_label.config(text=f"Renamed File: {new_file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to rename file: {str(e)}")

# Function to play video in the tkinter frame
def play_video(file_path):
    cap = cv2.VideoCapture(file_path)

    def stream():
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)
            video_label.after(10, stream)
        else:
            cap.release()

    stream()

# Function to process video with YOLO and OCR
def process_video():
    if not selected_file:
        messagebox.showerror("Error", "No video file selected.")
        return

    cap = cv2.VideoCapture(selected_file)
    
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
            
            # Update best frame if this frame has highest confidence detection
            if max_conf > best_confidence:
                best_confidence = max_conf
                best_frame = frame
                best_box = current_boxes[max_conf_idx]
    
    # Save the best frame with the highest confidence score
    if best_frame is not None:
        save_path = os.path.join(save_dir, f"best_frame.jpg")
        cv2.imwrite(save_path, best_frame)
        print(f"Saved the best frame with the highest confidence score at: {save_path}")
        print(f"Bounding Box: {best_box}, Confidence: {best_confidence}")
    else:
        print("No suitable frame found.")
        messagebox.showerror("Error", "No suitable frame found.")
        return

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=True)

    # Process the best frame with EasyOCR
    result = reader.readtext(save_path)
    
    # Variables to store positions of Scene, Take, Roll, and numbers
    scene_position = None
    roll_position = None
    number_positions = []
    
    # Iterate through OCR detections
    for detection in result:
        box, text, _ = detection
        center = calculate_center(box)

        # Identify 'SCENE', 'ROLL', 'TAKE' and numbers
        if 'SCENE' in text.upper():
            scene_position = center
        elif 'ROLL' in text.upper():
            roll_position = center
        elif text.isdigit():
            number_positions.append((center, text))

    # Find nearest number to Scene and Roll
    scene_value = None
    roll_value = None

    # Check if any numbers were detected before calculating the nearest value
    if scene_position is not None and number_positions:
        scene_value = min(number_positions, key=lambda x: calculate_distance(scene_position, x[0]))[1]
    else:
        scene_value = "N/A"

    if roll_position is not None and number_positions:
        roll_value = min(number_positions, key=lambda x: calculate_distance(roll_position, x[0]))[1]
    else:
        roll_value = "N/A"

    # Show the results in the GUI
    scene_value = f"Scene: {scene_value}" if scene_value else "Scene: N/A"
    roll_value = f"Roll: {roll_value}" if roll_value else "Roll: N/A"
    messagebox.showinfo("Detection Results", f"{scene_value}\n{roll_value}")

    # Display the frame with matplotlib
    img = cv2.imread(save_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Create the main window
root = tk.Tk()
root.title("MP4 Video Preview and Process")

# Create and place widgets
open_button = tk.Button(root, text="Open MP4 File", command=open_file)
open_button.pack(pady=10)

file_label = tk.Label(root, text="No file selected")
file_label.pack()

entry_label = tk.Label(root, text="Enter New Name:")
entry_label.pack()

entry = tk.Entry(root)
entry.pack(pady=5)

rename_button = tk.Button(root, text="Rename File", command=rename_file)
rename_button.pack(pady=10)

process_button = tk.Button(root, text="Process Video", command=process_video)
process_button.pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

# Initialize selected_file as None
selected_file = None

# Run the Tkinter event loop
root.mainloop()

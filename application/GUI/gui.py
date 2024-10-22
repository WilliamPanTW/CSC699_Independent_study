import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Create the main window
root = tk.Tk()
root.title("MP4 Video Preview and Rename")

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

    # Get the file extension and new file path
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
            # Convert the frame to RGB (from BGR as OpenCV uses BGR)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)
            # Call the function again after a delay (loop)
            video_label.after(10, stream)
        else:
            cap.release()

    stream()

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

video_label = tk.Label(root)
video_label.pack()

# Initialize selected_file as None
selected_file = None

# Run the Tkinter event loop
root.mainloop()

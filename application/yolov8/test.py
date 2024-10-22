from ultralytics import YOLO
import os 

# Load a model
model = YOLO(r'D:\CSC699_Independent_study\application\yolov8\yolov8_custom.pt') # pretrained YOLOv8n model

results = model.predict(
    source=r"D:\CSC699_Independent_study\application\yolov8\SLATE.mp4",  # Input video
    show=True,   # Display the results
    save=True,   # Save the results
    save_dir=r"D:\CSC699_Independent_study\application\yolov8\results",  # Save to this directory
    conf=0.5     # Confidence threshold
)
print(f"Results saved to: {os.path.abspath(results[0].save_dir)}")

# yolo task=detect mode=predict model=yolov8_custom.pt show=true conf=0.5 source=top2.jpg
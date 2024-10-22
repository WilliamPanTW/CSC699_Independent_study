from ultralytics import YOLO

model= YOLO("yolov8m.pt")

model.train(data="data_custom.yaml" ,imgsz=640,epochs=100,workers=0)
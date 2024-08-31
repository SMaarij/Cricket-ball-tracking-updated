from ultralytics import YOLO
import os

# Define the path to your data.yaml file
data_path = 'C:/Users/NEDUET/Desktop/CricketBallDetection.v1i.yolov8-obb/Cricket-Ball-Tracking-main/data.yaml'

# Verify the path
if not os.path.isfile(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

# Load a pre-trained YOLOv8 model (adjust the model type if necessary)
model = YOLO('yolov8n.pt')

# Train the model
model.train(
    data=data_path,       # Path to your dataset YAML file
    epochs=200,           # Number of epochs for training
    imgsz=640,           # Image size (640x640)
    batch=50            # Batch size (adjust as needed)
)

print("Training complete.")

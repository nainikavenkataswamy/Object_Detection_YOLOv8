# Object Detection in Image Using YOLOV8

import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model (choose 'yolov8n.pt', 'yolov8s.pt', etc. for different sizes)
model = YOLO('./runs/detect/train4/weights/best.pt')  # or another version of YOLOv8

# User provides the image path
input_image_path = './test_images/4.jpg'  # Change this to your image path
output_image_path = './test_image_outputs/output_image1.jpg'  # Output path for the processed image

# Read the image using OpenCV
image = cv2.imread(input_image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not read the image.")
else:
    # Apply YOLOv8 object detection
    results = model(image)[0]

    # Iterate through the detections and draw bounding boxes
    for result in results.boxes.data.tolist():  # Each detection in the format [x1, y1, x2, y2, conf, class]
        x1, y1, x2, y2, conf, cls = result[:6]
        label = f'{model.names[int(cls)]} {conf:.2f}'
        
        # Draw bounding box and label on the image
        if conf > 0.5: 
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Bounding box
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the processed image
    cv2.imwrite(output_image_path, image)

    print(f'Output image saved to {output_image_path}')

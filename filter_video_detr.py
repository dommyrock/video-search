from transformers import DetrForObjectDetection, DetrFeatureExtractor
import torch
import cv2
import numpy as np

# requirems
# pip install timm`

# Initialize the DETR model and feature extractor
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

# Move the model to the GPU
model = model.to('cuda')

# Load your video
cap = cv2.VideoCapture('5_min_vid.mp4')

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare the frame for the model
    inputs = feature_extractor(images=frame_rgb, return_tensors="pt")

    # Move the inputs to the GPU
    inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}

    # Run the frame through the model
    outputs = model(**inputs)

    # Get the predicted labels
    # Note: You'll need to write additional code to filter out the faces from the video frames
    # based on the predicted labels

cap.release()
cv2.destroyAllWindows()

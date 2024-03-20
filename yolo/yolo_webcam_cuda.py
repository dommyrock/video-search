# AssertionError: Torch not compiled with CUDA enabled
# https://pytorch.org/get-started/locally/ (will show you "Run this Command"  <script>)
# https://stackoverflow.com/questions/57814535/assertionerror-torch-not-compiled-with-cuda-enabled-in-spite-upgrading-to-cud
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# SEEMS LIKE ITS NOT CURRENTLY POSSIBLE LOCALLY : https://github.com/ultralytics/ultralytics/issues/3084
# but could be if combined with onnx runtime 

import cv2
from ultralytics import YOLO
import torch

#setup gpu
device="cuda" if  torch.cuda.is_available() else "cpu"

if device == "cuda":
   torch.cuda.set_device(0)

print(f"Device: {device}\n")

# Load the YOLOv8 model
model = YOLO("yolov8n.pt").to(device)


# Run inference on the webcam source
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        # results = model.track(frame, persist=True) #threw torch.cuda.is_available(): False

        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # End of the video is reached
        break

cap.release()
cv2.destroyAllWindows()

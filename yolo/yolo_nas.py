# pip install super_gradients (also requires cmake https://cmake.org/download/)
# pip install ultralytics
# pip install opencv-python

import cv2
from ultralytics import NAS

model = NAS("yolo_nas_s.pt")

# Display model information (optional)
model.info()

# Open the video source
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Only track 'person' class
        results = model.track(frame, classes=[0], persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # End of the video is reached
        break

cap.release()
cv2.destroyAllWindows()

#other models 
# Model              mAP      Latency(ms)
# YOLO-NAS S 	      47.5 	   3.21
# YOLO-NAS M 	      51.55 	5.85
# YOLO-NAS L 	      52.22 	7.87
# YOLO-NAS S INT-8 	47.03 	2.36
# YOLO-NAS M INT-8 	51.0 	   3.78
# YOLO-NAS L INT-8 	52.1 	   4.78
# https://docs.ultralytics.com/models/yolo-nas/#key-features
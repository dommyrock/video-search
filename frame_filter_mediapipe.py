import cv2
import mediapipe as mp
import time
import sys

# Check if a filename was provided
if len(sys.argv) < 2:
    print("Usage: python script.py <video_file>")
    sys.exit()

# Get the video filename from the command-line arguments
video_file = sys.argv[1]

# Create instances of the FaceDetection and DrawingUtils classes
face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# Open the video file
cap = cv2.VideoCapture(video_file)

# Get the video's frames per second and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('edit_facetrim.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    success, image = cap.read()
    start = time.time()

    if not success:
        print("Ignoring empty camera frame.")
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, mark the image as not writeable to pass by reference
    image.flags.writeable = False
    results = face_detection.process(image)

    # If no faces are detected, write the frame to the new video file
    if not results.detections:
        # Convert the RGB image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)

    end = time.time()
    totalTime = end - start

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
out.release()

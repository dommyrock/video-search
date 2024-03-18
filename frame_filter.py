# OpenCV’s built-in Haar cascades - face detection
import cv2
import sys

# Check if a filename was provided
if len(sys.argv) < 2:
    print("Usage: python script.py <video_file>")
    sys.exit()

# Get the video filename from the command-line arguments
video_file = sys.argv[1]

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video file
video = cv2.VideoCapture(video_file)

# Get the video's frame size and frames per second
# The video.get(3) and video.get(4) calls are getting the width and height of the video frames, respectively, directly from the video file. This means that the script will adapt to videos of different resolutions without any modifications. So, if your video is always at least 720p, this part of the code will handle it correctly.
frame_width = int(video.get(3))
frame_height = int(video.get(4))
fps = video.get(cv2.CAP_PROP_FPS)

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('edit_facetrim.mp4', fourcc, fps, (frame_width, frame_height))

# Process each frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale (required for Haar cascades)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, write the frame to the new video file
    if len(faces) == 0:
        out.write(frame)

# Release the video capture and writer objects
video.release()
out.release()

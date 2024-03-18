import cv2
import csv
import sys

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if a filename was provided
if len(sys.argv) < 2:
    print("Usage: python script.py <video_file>")
    sys.exit()

# Get the video filename from the command-line arguments
video_file = sys.argv[1]

# Open the video file
video = cv2.VideoCapture(video_file)

# Get the video's frames per second
fps = video.get(cv2.CAP_PROP_FPS)

# Initialize the frame count and timestamps list
frame_count = 0
timestamps = []
unique_seconds = set()

# Process each frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale (required for Haar cascades)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If a face is detected, increment the frame count and save the timestamp
    if len(faces) > 0:
        frame_count += 1
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = frame_number / fps
        timestamps.append(timestamp)
        unique_seconds.add(round(timestamp))

# Release the video file
video.release()

# Print the total frame count
print(f'Frames with faces frame count: {frame_count}')

if frame_count > 0:
   # Write the timestamps to a CSV file
   csv_file = video_file.split('.')[0] + '_face_timestamps.csv'
   with open(csv_file, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(['Timestamp'])
      for timestamp in timestamps:
         writer.writerow([timestamp])

   # Write the unique seconds to another CSV file
   csv_file_seconds = video_file.split('.')[0] + '_face_timestamps_seconds.csv'
   with open(csv_file_seconds, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(['Seconds with faces'])
      for second in sorted(unique_seconds):
         writer.writerow([second])

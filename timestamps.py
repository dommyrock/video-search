import cv2
import csv
import sys
import mediapipe as mp
import time

# Create instances of the FaceDetection
face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

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

print('Processing video ...')
start_time = time.time()

# Process each frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to RGB (required for mediapipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance, mark the image as not writeable to pass by reference
    rgb.flags.writeable = False
    results = face_detection.process(rgb)

    # If a face is detected, increment the frame count and save the timestamp
    if results.detections:
        frame_count += 1
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = frame_number / fps
        timestamps.append(timestamp)
        unique_seconds.add(round(timestamp))

# Release the video file
video.release()

# Stop the timer and calculate the total execution time
end_time = time.time()
total_time = end_time - start_time
print(f'Total execution time: {total_time} seconds')


# Print the total frame count
print(f'Total faces frame count: {frame_count}')

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

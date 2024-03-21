import cv2
import mediapipe as mp
import time
import sys
import subprocess
import shutil

# Check if a filename was provided
if len(sys.argv) < 2:
    print("Usage: python script.py <video_file>")
    sys.exit()

# Get the video filename from the command-line arguments
video_file = sys.argv[1]

# Create instances of the FaceDetection and DrawingUtils classes
face_detection = mp.solutions.face_detection.FaceDetection( model_selection=1, min_detection_confidence=0.3)

# Open the video file
cap = cv2.VideoCapture(video_file)

# Get the video's frames per second and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('edit_facetrim.mp4', fourcc, fps, (frame_width, frame_height))

start = time.time()
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Reached empty camera frame.  Exiting...")
        break

    # Convert the BGR image to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance, mark the image as not writeable to pass by reference
    frame.flags.writeable = False
    results = face_detection.process(frame)

    # If no faces are detected, write the frame to the new video file
    if not results.detections:
        # Convert the RGB image back to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
out.release()
end = time.time()

# Convert the total time to minutes and seconds
totalTime = end - start
minutes, seconds = divmod(totalTime, 60)
print(f"Execution time: {minutes:.0f}:{seconds:.2f}")

print("Compressing the edit_facetrim.mp4 ...")

# Check if ffmpeg is installed
if shutil.which("ffmpeg") is None:
    print("ffmpeg is not installed. You can download it from https://www.ffmpeg.org/download.html")
    sys.exit()
    

# Call the ffmpeg compression script on the output video
subprocess.run(["ffmpeg", "-i", "edit_facetrim.mp4", "-vcodec", "libx264", "-crf", "23", "compressed_edit_facetrim.mp4"])

#mostly used for vision model validations 
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

video_file = sys.argv[1]

mp_drawing = mp.solutions.drawing_utils
face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.3)

# Open the video file
cap = cv2.VideoCapture(video_file)

# Get the video's frames per second and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('bounding_box_edit.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    success, image = cap.read()
    start = time.time()

    if not success:
        print("Reached empty camera frame.  Exiting...")
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, mark the image as not writeable to pass by reference
    image.flags.writeable = False
    results = face_detection.process(image)

    # Convert the RGB image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the face detections on the image
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)

    # Write the frame to the new video file
    out.write(image)

    end = time.time()
    totalTime = end - start

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
out.release()


#TEST Compression if needed

# print("Compressing the edit_facetrim.mp4 ...")

# # Check if ffmpeg is installed
# if shutil.which("ffmpeg") is None:
#     print("ffmpeg is not installed. You can download it from https://www.ffmpeg.org/download.html")
#     sys.exit()
    

# # Call the ffmpeg compression script on the output video
# subprocess.run(["ffmpeg", "-i", "edit_facetrim.mp4", "-vcodec", "libx264", "-crf", "23", "compressed_edit_facetrim.mp4"])

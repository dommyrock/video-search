import cv2
import sys
from pydub import AudioSegment

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
frame_width = int(video.get(3))
frame_height = int(video.get(4))
fps = video.get(cv2.CAP_PROP_FPS)

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Load the audio file
audio = AudioSegment.from_file('input.mp4')

# Process each frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale (required for Haar cascades)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Get the current frame number
    frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))

    # Calculate the corresponding time in the audio file
    audio_time = frame_number / fps * 1000  # pydub works in milliseconds

    # Check if there's audio at this time
    audio_chunk = audio[audio_time:audio_time+1000]  # 1000 ms = 1 second
    if audio_chunk.rms > 1000:  # adjust this value based on your audio
        has_audio = True
    else:
        has_audio = False

    # If a face is detected and there's audio, skip this frame
    if len(faces) > 0 and has_audio:
        continue

    # Otherwise, write the frame to the new video file
    out.write(frame)

# Release the video capture and writer objects
video.release()
out.release()

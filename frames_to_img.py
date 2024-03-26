import cv2
import os
import time

# Create the directory if it doesn't exist
if not os.path.exists('frames'):
    os.makedirs('frames')

# Open the video file
cap = cv2.VideoCapture('1_edit.mp4')

# Initialize frame counter
cnt = 0

# Start timing the execution
start_time = time.time()

while(cap.isOpened()):
    # Read the frame from the video
    ret, frame = cap.read()

    if ret == True:
        # Write the results to a png file
        cv2.imwrite('./frames/frame{}.png'.format(cnt), frame)
        cnt += 1
    else:
        break

# Release the VideoCapture object
cap.release()

# Stop timing the execution
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print("Execution time: {:.4f} seconds".format(execution_time))
print(f'total frames: {cnt}')

# Get the size of the frames directory
if  os.path.exists('frames'):
  frames_dir = os.path.abspath('frames')
  frames_size_bytes = sum(os.path.getsize(os.path.join(frames_dir, f)) for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f)))

  frames_size_mb = frames_size_bytes / 1024**2

  # Print the size of the frames directory
  print(f'/frames directory size: {frames_size_mb:.2f} MB')
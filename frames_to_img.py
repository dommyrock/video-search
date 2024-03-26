import cv2
import os
import time
import shutil

# Create / reset the 'out' directory 
frames_dir = 'frames'
try:
    shutil.rmtree(frames_dir)
except OSError:
    pass
os.makedirs(frames_dir)

# Open the video file
cap = cv2.VideoCapture('1_edit.mp4')

total_fame_cnt = 0
saved_frame_cnt = 0
fame_skip_interval = 10

start_time = time.time()

while(cap.isOpened()):
    # Read the frame from the video
    success, frame = cap.read()

    if success:
        if total_fame_cnt % fame_skip_interval == 0:
            cv2.imwrite('./frames/frame{}.jpg'.format(saved_frame_cnt), frame,[cv2.IMWRITE_JPEG_QUALITY, 80])
            saved_frame_cnt +=1
        total_fame_cnt += 1
    else:
        break

cap.release()
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print("Execution time: {:.4f} seconds".format(execution_time))
print(f'total frames: {total_fame_cnt} | saved frames: {saved_frame_cnt}')

# Get the size of the frames directory
if  os.path.exists('frames'):
  frames_dir = os.path.abspath('frames')
  frames_size_bytes = sum(os.path.getsize(os.path.join(frames_dir, f)) for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f)))

  frames_size_mb = frames_size_bytes / 1024**2
  print(f'/frames directory size: {frames_size_mb:.2f} MB')
import cv2
import sys

# Check if a filename was provided
if len(sys.argv) < 2:
    print("Usage: python script.py <video_file1.mp4> <video_file2.mp4>")
    sys.exit()

# Get the video filenames from the command-line arguments
video_file1 = sys.argv[1]
video_file2= sys.argv[2]

# Open the video files
video1 = cv2.VideoCapture(video_file1)
video2 = cv2.VideoCapture(video_file2)

# Get the frame count
frame_count1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))

# Get the frame rate (fps)
fps1 = video1.get(cv2.CAP_PROP_FPS)
fps2 = video2.get(cv2.CAP_PROP_FPS)

# Get the frame size
frame_size1 = (int(video1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
frame_size2 = (int(video2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Print the statistics
print(f'Video file: {video_file1}')
print(f'Frame count: {frame_count1}')
print(f'Frame rate: {fps1}')
print(f'Frame size: {frame_size1}')

print(f'\nVideo file: {video_file2}')
print(f'Frame count: {frame_count2}')
print(f'Frame rate: {fps2}')
print(f'Frame size: {frame_size2}')

# Release the video files
video1.release()
video2.release()

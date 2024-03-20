# pip install ultralytics
# python yolo_frame_filter_person.py ../low_quality_vid.mp4

import cv2
import sys
import time
from ultralytics import YOLO

if len(sys.argv) < 2:
    print("Usage: python script.py <video_file>")
    sys.exit()

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

# Get the video's frames per second and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'XVID'
out = cv2.VideoWriter("edit_facetrim.mp4", fourcc, fps, (frame_width, frame_height))

start = time.time()
# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Reached empty camera frame.  Exiting...")
        break

    results = model(frame)
    person_detections = []
    # Convert the BGR image to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for result in results:
        for box in result.boxes:  # Iterate through the boxes in the current result
            class_id = box.cls[0].item()  # Get the class ID / 0 = 'person'
            conf = box.conf[0].item()  # Get the confidence

            # Filter based on the class ID and confidence
            if class_id == 0 and conf >= 0.75:
                person_detections.append((box, conf))

    # Write the frame to edited video frames ONLY IF NO cls[0] ='person' is detected
    if not person_detections:
        # Convert the RGB image back to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Visualize the results on the frame, ONLY where "person" is detected
    #  for box, conf in person_detections:
    #      x1, y1, x2, y2 = map(int, box.xyxy[0])
    #      cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #      cv2.putText(
    #          frame,
    #          f"{conf:.2f}",
    #          (x1, y1 - 10),
    #          cv2.FONT_HERSHEY_SIMPLEX,
    #          0.5,
    #          (255, 0, 0),
    #          2,
    #      )

    #  # Display the annotated frame
    #  cv2.imshow("YOLOv8 Inference", frame)

cap.release()
out.release()
# cv2.destroyAllWindows()  # remove when done testing
end = time.time()

# Convert the total time to minutes and seconds
totalTime = end - start
minutes, seconds = divmod(totalTime, 60)
print(f"Execution time: {minutes:.0f}:{seconds:.2f}")

# pip install ultralytics
import cv2
import sys
from ultralytics import YOLO

# displays confidance levels next to bounding box

if len(sys.argv) < 2:
    print("Usage: python script.py <video_file>")
    sys.exit()
    
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

# Loop through the video frames

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)
        person_detections = []

        for result in results:  # Iterate through the results (there could be multiple objects detected)
            for box in result.boxes:  # Iterate through the boxes in the current result
                class_id = box.cls[0].item()  # Get the class ID
                conf = box.conf[0].item()  # Get the confidence

                if class_id == 0 and conf >= 0.5:  # Filter based on the class ID and confidence
                    person_detections.append((box, conf))

        # Visualize the results on the frame, ONLY where "person" is detected
        for box, conf in person_detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
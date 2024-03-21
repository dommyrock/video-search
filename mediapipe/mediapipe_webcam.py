import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) # 0 = webcam input
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.3) as face_detection:
  # model_selection=1 =Long range sparse
  #https://github.com/google/mediapipe/blob/master/docs/solutions/face_detection.md#model_selection
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to pas by ref
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame)

    # Draw the face detection annotations on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(frame, detection)
        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(frame, 1))
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release() 
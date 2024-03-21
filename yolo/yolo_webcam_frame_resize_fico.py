import cv2
from ultralytics import YOLO
import torch

def select_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)
    return device

def load_model():
    device = select_device()
    model = YOLO('yolov8n.pt').to(device)
    return model, device

def set_window_size(window_name, capture):
    video_width = int(capture.get(3))
    video_height = int(capture.get(4))
    video_aspect_ratio = video_width / video_height

    window_width = cv2.getWindowImageRect(window_name)[2]
    window_height = cv2.getWindowImageRect(window_name)[3]
    window_aspect_ratio = window_width / window_height

    if window_aspect_ratio > video_aspect_ratio:
        cv2.resizeWindow(window_name, int(window_height * video_aspect_ratio), window_height)
    if window_aspect_ratio < video_aspect_ratio:
        cv2.resizeWindow(window_name, window_width, int(window_width / video_aspect_ratio))

def should_break(window_name, cv2):
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return True
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        return True
    return False

def run_inference(model, device):
    cap = cv2.VideoCapture(0)
    window_name = "YOLOv8 Inference"
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        inverted_frame = cv2.flip(frame, 1)
        results = model.track(inverted_frame, persist=True)
        annotated_frame = results[0].plot()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, annotated_frame)
        set_window_size(window_name, cap)

        if (should_break(window_name, cv2)):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    model, device = load_model()
    print(f"Device: {device}\n")
    run_inference(model, device)

if __name__ == "__main__":
    main()
# pip install sahi ultralytics
# run
# python yolo_v8_sahi_person.py --source "webcam" --save-img --view-img
# python yolo_v8_sahi_person.py --source "webcam" --view-img

import argparse
from pathlib import Path
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model
from ultralytics.utils.files import increment_path


def run(
    weights="yolov8n.pt",
    source="test.mp4",
    view_img=False,
    save_img=False,
    exist_ok=False,
):
    """
    Run object detection on a video using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
    """

    # Check source path
    if not (source == "webcam" or Path(source).exists()):
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    yolov8_model_path = f"models/{weights}"
    # download_yolov8s_model(yolov8_model_path)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=yolov8_model_path,
        confidence_threshold=0.3,
        device="cpu",
    )

    # Video setup
    videocapture = cv2.VideoCapture(0)  # source / or 0 = webcam
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    if save_img:
        save_dir = increment_path(
            Path("ultralytics_results_with_sahi") / "exp", exist_ok
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            str(save_dir / f"{Path(source).stem}.mp4"),
            fourcc,
            fps,
            (frame_width, frame_height),
        )

    while videocapture.isOpened():
        success, frame = videocapture.read()

        if not success:
            break

        # 'SAHI' partitions images into manageable slices,
        # performs object detection on each slice, and then stitches the results back together.
        results = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        object_prediction_list = results.object_prediction_list

        # Filter the 'person' class only
        person_boxes_list = []
        person_clss_list = []
        for obj_pred in object_prediction_list:
            if obj_pred.category.name == "person":
                boxes = (
                    obj_pred.bbox.minx,
                    obj_pred.bbox.miny,
                    obj_pred.bbox.maxx,
                    obj_pred.bbox.maxy,
                )
                person_boxes_list.append(boxes)
                person_clss_list.append("person")

        for box, cls in zip(person_boxes_list, person_clss_list):
            x1, y1, x2, y2 = box
            cv2.rectangle(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2
            )
            label = str(cls)
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]

            cv2.rectangle(
                frame,
                (int(x1), int(y1) - t_size[1] - 3),
                (int(x1) + t_size[0], int(y1) + 3),
                (56, 56, 255),
                -1,
            )
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 2),
                0,
                0.6,
                [255, 255, 255],
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        if view_img:
            cv2.imshow(Path(source).stem, frame)

        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if save_img:
        video_writer.release()

    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="yolov8n.pt", help="initial weights path"
    )
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    return parser.parse_args()


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

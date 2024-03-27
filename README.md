
### Script usage

- **video_download.py**

Script used for downloading video sources.

## YOLOv8 scripts 
Can be found in **/yolo** and **/yolo_sahi** directories 

yolov8 models mostly require these deps
```bash
pip install opencv-python
pip install ultralytics
pip install sahi

```

Great post:<br/>
[YOLOv8 object detection ](https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/)

[Ultralythics](https://docs.ultralytics.com/)
[Ultralythics repo](https://github.com/ultralytics/ultralytics)
[Supervision lib repo](https://github.com/roboflow/supervision)

[Ultralytics models](https://docs.ultralytics.com/models/#featured-models)

[YOLOV8 face detection PT weights examples](https://github.com/akanametov/yolov8-face/blob/dev/examples/tutorial.ipynb)

Running Yolov8 on GPU issues
- https://github.com/ultralytics/ultralytics/issues/3084
- https://github.com/ultralytics/ultralytics/issues/5059

[My GPU example](https://github.com/dommyrock/video-search/blob/main/yolo/yolo_webcam_cuda.py)

### Similarity search and frame filtering 
   Has to be done on linux env since 'faiss-gpu' from 'pip install transformers faiss-gpu torch Pillow' fails on windows
   
   [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)

   Examples --> **/similarity** (both with **torchvision** and **transformers** API)

   > Note: 'opencv-python' also had few other requirements after pip install 'opencv-python' > [Open Cv Requirements](https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo)
   
   In my case this cmd helped on linux
   ```bash
   sudo apt-get update && sudo apt-get install -y python3-opencv
   ```
   Refs:
   https://medium.com/aimonks/image-similarity-with-dinov2-and-faiss-741744bc5804

OpenCV vs Torchvision:

OpenCV (Open Source Computer Vision Library):

- OpenCV is a highly optimized library with a focus on real-time applications.
- It supports a wide variety of image and video file formats.
- OpenCV provides functions for image and video processing such as filtering, geometric transformations, color space manipulation, object detection, and much more.
It also includes algorithms for machine learning, computer vision, and artificial intelligence.
- OpenCV is compatible with many languages such as Python, C++, and Java, and can be used on different platforms including Windows, Linux, OS X, Android, and iOS.

Torchvision:
- Torchvision is a part of the PyTorch project, designed specifically for computer vision tasks.
- It provides utilities for efficiently loading and preprocessing image and video datasets, which can be very useful for training machine learning models.
- Torchvision also includes a number of pre-trained models for tasks such as image classification, object detection, and semantic segmentation.
- If you’re already using PyTorch for your project, using Torchvision can make a lot of sense because of the seamless integration between the two.

> In the context of processing video frames, both libraries can be suitable. If you’re simply looking to extract frames from a video, apply some basic transformations, and save the results, OpenCV might be the more straightforward choice. If you’re planning to use the frames as input to a machine learning model, especially one built with PyTorch, Torchvision could be more convenient.

## Mediapipe 

- **frame_filter_mediapipe.py** (MOST EFFICIENT ONE AT THE MOMENT)

A lot less heavy on the cpu and is faster.<br/>
Uses "mediapipe" https://developers.google.com/mediapipe/solutions/vision/face_detector

```bash
# pip install opencv-python
# pip install mediapipe

python frame_filter_mediapipe.py <video.mp4>
```

- **timestamps.py**

processes video such that it counts total frames with faces in them.<br/>
Outputs 2x files with ALL timestams and grouped by seconds (if there is any face in [frames] of that second)<br/>
There are other scripts taht use Haar algo for face detection:timestamps_haar.py, ts_all_haar.py.

```bash
python timestamps.py <video.mp4>
#or
python ts_all_haargit init.py <video.mp4>
```

--- 
Other scripts 

- **frame_filter.py**
Script used to filter out Faces from video.<br/>
Uses pre-trained Haar cascade for face detection.

- **frame_filter_audio_dub.py**

extension of above filtering scriot to also include audio detection in combination wiht facee detection.<br/>
Gives us a chance to confirm we should delete the frame in a more precise way ie. when person is detected in a freame and also audio confirming it.

- **compare.py**

Extracts video metadata comparing 2 videos.

```bash
python compare.py <video1>.mp4 <video2>.mp4
```

```python
# Prints the statistics
print(f'Video file: {video_file1}')
print(f'Frame count: {frame_count1}')
print(f'Frame rate: {fps1}')
print(f'Frame size: {frame_size1}')
```


```bash
pip install opencv-python
pip install pydub # https://github.com/jiaaro/pydub
```

### compression ffmpeg (currently called from **frame_filter_mediapipe.py** script)

```bash
ffmpeg -i <input>.mp4 -vcodec libx264 -crf 23 <output>.mp4

```

### NOTES: 
Some faster alternatives : 

[Streams api](https://pytube.io/en/latest/user/streams.html#downloading-streams) 

[what are the steps to run the above python script on the windows os  ?](https://stackoverflow.com/questions/67521143/how-to-make-code-run-on-gpu-on-windows-10)

```bash
conda create --name tf_GPU tensorflow-gpu
conda activate tf_GPU

#add python scrip deps if needed

#run script 
python script.py

```
https://developer.nvidia.com/how-to-cuda-python


### Testing: 
See this vid https://www.youtube.com/watch?v=YpCcAyxj-Vg <br/>
It didn't cut out few quite obvious frames for some reason

#### bash utils
Move all files from current pwd to /yolo directory
```bash
find . -maxdepth 1 -type f -name "yolo_*" -exec mv {} yolo/ \;
```

Copy local windows dir to pwd in wsl2 Ubuntu
```bash
cp -r /mnt/c/Users/dpolzer/Me/Git/video-search/frames .
```

Check img file count inside "frames" dir
```bash
ls -1 frames | wc -l
# exclude nested directories if ther are any
ls -1 frames | grep -v / | wc -l
```

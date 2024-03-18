
### Script usage

- **video_download.py**

Script used for downloading video sources.

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

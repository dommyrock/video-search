# You can use `ffmpeg`, a powerful tool that can handle a variety of multimedia tasks including extracting audio, converting video formats, and more. Here's how you can use `ffmpeg` to add the audio from the original video to the output video:
# 
# 1. **Extract the audio from the original video**:
# ```bash
# ffmpeg -i input.mp4 -vn -acodec copy audio.aac
# ```
# This command extracts the audio from 'input.mp4' and saves it as 'audio.aac'. You can change these filenames to whatever you want.
# 
# 2. **Add the audio to the output video**:
# ```bash
# ffmpeg -i output.mp4 -i audio.aac -c:v copy -c:a aac final_output.mp4
# ```
# This command takes the video from 'output.mp4' and the audio from 'audio.aac', and writes them to 'final_output.mp4'. Again, you can change these filenames to whatever you want.
# 
# Please note that you'll need to have `ffmpeg` installed on your machine to use these commands. You can download it from the [official website](https://ffmpeg.org/download.html). Also, these commands are for a Unix-like command line (like Linux or MacOS). If you're using Windows, you might need to adjust them slightly. Good luck with your project!
 

#todo  - Test this out 
ffmpeg -i input.mp4 -vn -acodec copy audio.aac

ffmpeg -i output.mp4 -i audio.aac -c:v copy -c:a aac final_output.mp4


#other option 
# from moviepy.editor import VideoFileClip
# 
# # Load the original video and the output video
# original = VideoFileClip('input.mp4')
# output = VideoFileClip('output.mp4')
# 
# # Set the audio of the output video to the audio of the original video
# output = output.set_audio(original.audio)
# 
# # Write the result to a file
# output.write_videofile('final_output.mp4')

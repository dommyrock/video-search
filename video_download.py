from pytube import YouTube #https://pytube.io/en/latest/
import sys

# Check if a filename was provided
if len(sys.argv) < 2:
    print("Usage: python script.py <video_file>")
    sys.exit()

# Get the video filename from the command-line arguments
href = sys.argv[1]

yt = YouTube(href)

# extact video metadata
title = yt.title
vid_len = round(yt.length / 60,2) # min
views =yt.views
author= yt.author
date = yt.publish_date
kw = yt.keywords

print("Title: " + title)
print("Author: " + author)
print("Date: " + str(date))
print("Video len: " + str(vid_len) + "min")  # Convert vid_len to string
print("Views: " + str(views))
print("KW: " + str(kw))

# Filter the streams to get the highest resolution stream that includes both audio and video .get_highest_resolution()
stream = yt.streams.filter(progressive=True, file_extension='mp4').get_by_resolution("720p")

#or
# Filter the streams to get the highest resolution progressive stream
# stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

stream.download()
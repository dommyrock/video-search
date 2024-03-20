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
vid_len = round(yt.length / 60) # min
views =yt.views
author= yt.author
date = yt.publish_date
kw = yt.keywords

print("Title: " + title)
print("Author: " + author)
print("Date: " + str(date))
print("Video len: {}min {}sec".format(vid_len, yt.length % 60))
print("Views: " + str(views))
print("KW: " + str(kw))

# Filter the streams to get the highest resolution stream that includes both audio and video .get_highest_resolution()
stream = yt.streams.filter(progressive=True, file_extension='mp4').get_by_resolution("720p")

#or
# Filter the streams to get the highest resolution progressive stream
# stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

# or don't use streams and get the minimum quality (use it for filtering  and getting the metadata, than search on FHD vid)

stream.download()
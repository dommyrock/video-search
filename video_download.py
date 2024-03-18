# docs: https://pytube.io/en/latest/
from pytube import YouTube

# Create a YouTube object
yt = YouTube('https://www.youtube.com/watch?v=sssrfVJUdFk')

# Get the title and length of the video
title = yt.title
vid_len = yt.length / 60 # min
views =yt.views
author= yt.author
date = yt.publish_date
kw = yt.keywords

# Print the title and length
print("Title: " + title)
print("Author: " + author)
print("Date: " + str(date))
print("Video len: " + str(vid_len) + "min")  # Convert vid_len to string
print("Views: " + str(views))
print("KW: " + str(kw))

# Filter the streams to get the highest resolution stream that includes both audio and video .get_highest_resolution()
# Filter the streams to get the highest resolution progressive stream
# stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
stream = yt.streams.filter(progressive=True, file_extension='mp4').get_by_resolution("720p")

# Download the video
stream.download()

# 3:54 is where face happens  / 
# # where we should check besides frame trim face 
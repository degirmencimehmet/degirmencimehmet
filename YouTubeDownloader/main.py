import pytubefix

url = input("Enter YouTube URL: ")

path = ""

pytubefix.YouTube(url).streams.get_lowest_resolution().download(path)

# get_lowest_resolution - en düşük kalitede
# get_highest_resolution - en yüksek kalitede
# .
# ..
# ...

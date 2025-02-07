import cv2 as cv
import pafy

url = "https://www.youtube.com/watch?v=ID-yWIleGAM"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

capture = cv.VideoCapture(best.url)

while True:
    grabbed, frame = capture.read()
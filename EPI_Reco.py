import cv2 as cv

capture = cv2.VideoCapture("")

while True:
    ret, frame = capture.read()
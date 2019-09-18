import numpy as np
import cv2 as cv
import time, datetime
from matplotlib import pyplot as plt
import operator

#Parameters
cap = cv.VideoCapture('rtsp://admin:Asngw9210@192.168.1.101/Streaming/Channels/1')
frame_num=0
frame_sample=20

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
while(cap.isOpened()):
    start = time.time()
    frame_num=frame_num+1
    ret, frame = cap.read()

    half_frame = cv.resize(frame, (0,0), fx=0.25, fy=0.25) 
    cv.imshow('frame',half_frame)
    if frame_num%frame_sample == 0:
        filename = str(st) + "_" + str(frame_num) +  ".jpg"
        cv.imwrite(filename, frame)
        print("Img [{}] is captured!".format(filename))


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


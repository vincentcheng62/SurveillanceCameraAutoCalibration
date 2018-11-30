import numpy as np
import cv2 as cv
import time, datetime
from matplotlib import pyplot as plt
import operator

cap = cv.VideoCapture('rtsp://admin:h0940232@172.18.9.100/Streaming/Channels/1')
frame_num=0
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
while(cap.isOpened()):
    start = time.time()
    frame_num=frame_num+1
    ret, frame = cap.read()

    cv.imshow('frame',frame)
    if frame_num%100 == 0:
        filename = str(st) + "_" + str(frame_num) +  ".jpg"
        cv.imwrite(filename, frame)
        print("Img [{}] is captured!".format(filename))


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


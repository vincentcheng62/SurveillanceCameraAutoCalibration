import numpy as np
import cv2 as cv
import time, datetime
from matplotlib import pyplot as plt
import operator

# Define the rectangluar ROI of the red and white light here
# The camera must be rigidly mounted and not going to change viewing angle in the future
red_light_mask_x = 354
red_light_mask_y = 259
red_light_mask_w = 27
red_light_mask_h = 48

white_light_mask_x = 383
white_light_mask_y = 345
white_light_mask_w = 15
white_light_mask_h = 18

#Parameters
#cap = cv.VideoCapture('traffic_light.mp4')
cap = cv.VideoCapture('traffic_light_night.mp4')

while(cap.isOpened()):
    start = time.time()
    ret, frame = cap.read()

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY ) 

    # Chop the ROI
    frame_gray_red_light = frame_gray[red_light_mask_y: red_light_mask_y+red_light_mask_h, red_light_mask_x: red_light_mask_x+red_light_mask_w]
    frame_gray_white_light = frame_gray[white_light_mask_y: white_light_mask_y+white_light_mask_h, white_light_mask_x: white_light_mask_x+white_light_mask_w]

    # Calculate the ROI volume
    red_sum = np.sum(frame_gray_red_light)
    white_sum = np.sum(frame_gray_white_light)

    # The average graylevel in the ROI
    red_sum = 1.0*red_sum/(red_light_mask_w*red_light_mask_h)
    white_sum = 1.0*white_sum/(white_light_mask_w*white_light_mask_h)

    print("red average graylevel= ", red_sum, "  , white average gray level=", white_sum)

    # Since white light will blink at the beginning, only change to red if the difference is large
    if 1.0*red_sum/white_sum > 2:
        cv.putText(frame, "Red light ON!", (red_light_mask_x+100, red_light_mask_y+red_light_mask_h), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    else:
        cv.putText(frame, "White light ON!", (white_light_mask_x+100, white_light_mask_y+white_light_mask_h), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)        

    # Resize and display
    half_frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5) 
    cv.imshow('frame',half_frame)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

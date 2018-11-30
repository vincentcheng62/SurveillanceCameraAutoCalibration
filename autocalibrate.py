import numpy as np
import cv2 as cv
import time, datetime
from matplotlib import pyplot as plt
import operator

cap = cv.VideoCapture('rtsp://admin:h0940232@172.18.9.100/Streaming/Channels/1')
#cap = cv2.VideoCapture('rtsp://172.18.9.99/axis-media/media.amp')
#time.sleep(5)
#print(cv2.__version__)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
#fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
#Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by
#the background model. This parameter does not affect the background update.
bg_history_frame=500
fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=8, detectShadows=False)
#fgbg = cv.bgsegm.createBackgroundSubtractorGMG()t
detector = cv.SimpleBlobDetector_create()
connectivity = 4
min_thresh=800
max_thresh=10000

cv.namedWindow("frame")
cv.moveWindow("frame", 40,10)
cv.namedWindow("fgmask")
cv.moveWindow("fgmask", 720,10)
cv.namedWindow("axis")
cv.moveWindow("axis", 40,420)

line_db_need_to_collect=100 # set lower for debug purpose
line_db = []
contour_area_min=600

#line in form of y=ax+c , so a tuple (a, c)
#return (IsHavingIntersection, inter_x, inter_y)
def find_two_line_intersection(line1, line2):
    if line1[0] == line2[0]:
        #print("The line are parallel!")
        return (False, -1, -1)
    else:
        a = line1[0]
        c = line1[1]
        b = line2[0]
        d = line2[1]
        return (True, (d-c)/(a-b), (a*d-b*c)/(a-b))


frame_num=0
while(cap.isOpened() and len(line_db)<line_db_need_to_collect):
    frame_num = frame_num+1
    start = time.time()
    ret, frame = cap.read()
    #print("cap.read() took {} seconds.".format(time.time() - start))
    start = time.time()
    #cv2.putText(frame, str(datetime.datetime.now()), (210, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, 2)
    #time.sleep(0.055)
    fgmask = fgbg.apply(frame)

    # erosion followed by dilation. 
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    fgmask = cv.dilate(fgmask,kernel2,iterations = 1)

    # output = cv.connectedComponentsWithStats(fgmask, connectivity, cv.CV_32S)
    # for i in range(output[0]):
    #     if output[2][i][4] >= min_thresh and output[2][i][4] <= max_thresh:
    #         cv.rectangle(fgmask, (output[2][i][0], output[2][i][1]), (
    #             output[2][i][0] + output[2][i][2], output[2][i][1] + output[2][i][3]), (255, 255, 255), 2)
    # cv.imshow('detection', fgmask)

    #keypoints = detector.detect(fgmask)
    #im_with_keypoints = cv.drawKeypoints(fgmask, keypoints, np.array([]), (255,255,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    thrhd_value = 1
    ret,fg_cnt_fitline = cv.threshold(fgmask,thrhd_value,255,cv.THRESH_BINARY)
    rows,cols = fg_cnt_fitline.shape[:2]
    im2, contours,hierarchy = cv.findContours(fg_cnt_fitline,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    #print("Number of contours: {}".format(len(contours)))

    fg_cnt_fitline_display = cv.cvtColor(fg_cnt_fitline,cv.COLOR_GRAY2RGB)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > contour_area_min:
            rect = cv.minAreaRect(cnt)
            #rect_height = rect[1][1]
            #rect_width = rect[1][0]
            box = cv.boxPoints(rect)
            min_y = 99999
            min_x = 99999
            max_y = -99999
            max_x = -99999
            for pt in box:
                if pt[1] < min_y:
                    min_y = pt[1]
                if pt[1] > max_y:
                    max_y = pt[1]
                if pt[0] < min_x:
                    min_x = pt[0]
                if pt[0] > max_x:
                    max_x = pt[0]                    

            rect_height = max_y-min_y
            rect_width = max_x-min_x
            #Standing pedestrian must be tall rectangle
            if rect_height > rect_width*2.0:

                
                boxx = np.int0(box)
                fg_cnt_fitline_display = cv.drawContours(fg_cnt_fitline_display,[boxx],0,(0,0,255),2)



                [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)

                # y=ax+c, a tuple (a, c)
                line_in_slope_form = (vy/vx, y-(vy/vx)*x)
                if line_in_slope_form[0] != 0:
                    if frame_num > bg_history_frame:
                        line_db.append(line_in_slope_form)
                    
                    min_y_x = (min_y-line_in_slope_form[1])/line_in_slope_form[0]
                    max_y_x = (max_y-line_in_slope_form[1])/line_in_slope_form[0]
                    fg_cnt_fitline_display = cv.line(fg_cnt_fitline_display,(min_y_x,min_y),(max_y_x,max_y),(0,255,0),2)

    resized_frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5) 
    resized_fgmask = cv.resize(fgmask, (0,0), fx=0.5, fy=0.5) 
    resized_fitline = cv.resize(fg_cnt_fitline_display, (0,0), fx=0.5, fy=0.5) 
    #imgBoth = np.hstack((resized_frame,resized_fgmask))
    cv.imshow('frame',resized_frame)

    # f = plt.figure()
    # f.add_subplot(1,2, 1)
    # plt.imshow(resized_frame)
    # f.add_subplot(1,2, 2)
    # plt.imshow(resized_fgmask)
    # plt.show(block=True)

    cv.imshow('fgmask',resized_fgmask)
    cv.imshow('axis',resized_fitline)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

print("Collected over {} major axis of pedestrian blob, start calculation...".format(line_db_need_to_collect))

#Allocate the voting space
voting_space={}
for line1 in line_db:
    for line2 in line_db:
        ret = find_two_line_intersection(line1, line2)
        if ret[0] == True:
            int_coord = (int(ret[1]), int(ret[2]))
            if int_coord in voting_space:
                voting_space[int_coord] = voting_space[int_coord]+1
            else:
                voting_space[int_coord] = 1

vote_coord = max(voting_space.items(), key=operator.itemgetter(1))[0]
print("Finish voting the pixel level vanishing point, which is {}".format(vote_coord))


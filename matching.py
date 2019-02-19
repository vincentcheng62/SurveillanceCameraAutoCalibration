import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#cap = cv.VideoCapture('onbridge.MOV')
#ret, img1 = cap.read()
img1 = cv.imread('onbridge.bmp') # trainImage
#img2 = cv.imread('snapshot200.png') # trainImage
img2 = cv.imread('snapshot200_unity.png') # trainImage?
scaledown=10.0
resized_img1 = cv.resize(img1, (0,0), fx=(1/scaledown), fy=(1/scaledown)) 
resized_img2 = cv.resize(img2, (0,0), fx=(1/scaledown), fy=(1/scaledown)) 


# Initiate SIFT detector
orb = cv.ORB_create(nfeatures=2000, scaleFactor=1.1)
# find the keypoints and descriptors with SIFT
kp1 = orb.detect(resized_img1,None)
kp2 = orb.detect(resized_img2,None)

kp1, des1 = orb.compute(resized_img1, kp1)
kp2, des2 = orb.compute(resized_img2, kp2)

# FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = cv.DrawMatchesFlags_DEFAULT)
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
# plt.imshow(img3,),plt.show()

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
#img1p = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
#img2p = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
Howmany=15

# struct DrawMatchesFlags
# {
#     enum
#     {
#         DEFAULT = 0, // Output image matrix will be created (Mat::create),
#                      // i.e. existing memory of output image may be reused.
#                      // Two source images, matches, and single keypoints
#                      // will be drawn.
#                      // For each keypoint, only the center point will be
#                      // drawn (without a circle around the keypoint with the
#                      // keypoint size and orientation).
#         DRAW_OVER_OUTIMG = 1, // Output image matrix will not be
#                        // created (using Mat::create). Matches will be drawn
#                        // on existing content of output image.
#         NOT_DRAW_SINGLE_POINTS = 2, // Single keypoints will not be drawn.
#         DRAW_RICH_KEYPOINTS = 4 // For each keypoint, the circle around
#                        // keypoint with keypoint size and orientation will
#                        // be drawn.
#     };
# };
img3 = cv.drawMatches(resized_img1,kp1,resized_img2,kp2,matches[:Howmany],None, flags=2)
img4 = cv.drawMatches(resized_img1,kp1,resized_img2,kp2,matches[:Howmany],None, flags=0)
cv.imwrite("zzz3.bmp", img3)
cv.imwrite("zzz4.bmp", img4)
#cv.imwrite("onbridge.bmp", img1)
resized_frame3 = cv.resize(img3, (0,0), fx=(scaledown*0.4), fy=(scaledown*0.4)) 
resized_frame4 = cv.resize(img4, (0,0), fx=(scaledown*0.4), fy=(scaledown*0.4)) 
comb = np.vstack((resized_frame3, resized_frame4))
cv.imwrite("zzz_comb.bmp", comb)
#plt.imshow(img3),plt.show()
cv.namedWindow("window", cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty("window",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
cv.imshow("result", comb)
cv.waitKey(0)   
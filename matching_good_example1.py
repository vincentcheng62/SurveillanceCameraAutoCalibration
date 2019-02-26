import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
#cap = cv.VideoCapture('onbridge.MOV')
#ret, img1 = cap.read()
#img1 = cv.imread('onbridge.bmp') # trainImage
#img2 = cv.imread('snapshot200_unity.png') # trainImage?
img1 = cv.imread('IMG_7094_half.JPG') # trainImage
img2 = cv.imread('screen_1920x1080_4.png') # trainImage?

# imgsample = img1.copy()
# imgsample = cv.resize(imgsample, (0,0), fx=(0.15), fy=(0.15)) 
# gray = cv.cvtColor(imgsample,cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray,150,300,apertureSize = 3)
# cv.imshow("edge", edges)

#blur = cv.GaussianBlur(gray,(5,5),0)
# blur = cv.medianBlur(gray,7)
# th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,13,0)
# cv.imshow("adaptive_thd", th3)

# minLineLength = 10
# maxLineGap = 30
# lines = cv.HoughLinesP(edges,1,5.0*np.pi/180,100,minLineLength,maxLineGap)
# print(len(lines))
# for line in lines:
#     cv.line(imgsample,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,255,0),2)
# cv.imshow("hough", imgsample)

# lines = cv.HoughLines(edges,1,np.pi/180,150)
# for i in  range(0,lines.shape[0]):
#     rho,theta = lines[i,0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv.line(imgsample,(x1,y1),(x2,y2),(0,0,255),1)
# cv.imshow("hough", imgsample)


# imgsample2 = img1.copy()
# imgsample2 = cv.resize(imgsample2, (0,0), fx=(0.5), fy=(0.5)) 
# # Parameters
# # ddepth	Specifies output image depth. Defualt is CV_32FC1
# # dx	Order of derivative x, default is 1
# # dy	Order of derivative y, default is 1
# # ksize	Sobel kernel size , default is 3
# # out_dtype	Converted format for output, default is CV_8UC1
# # scale	Optional scale value for derivative values, default is 1
# # delta	Optional bias added to output, default is 0
# # borderType	Pixel extrapolation method, default is BORDER_DEFAULT

# ridge_filter = cv.ximgproc.RidgeDetectionFilter_create(ksize=3)
# ridges = ridge_filter.getRidgeFilteredImage(imgsample2)
# cv.imshow("ridges", ridges)

# imgsample3 = img1.copy()
# gray3 = cv.cvtColor(imgsample3,cv.COLOR_BGR2GRAY)
# gray3 = cv.resize(gray3, (0,0), fx=(0.5), fy=(0.5)) 

# def detect_ridges(gray, sigma=1.0):
#     hxx, hyy, hxy = hessian_matrix(gray, sigma)
#     i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
#     return i1, i2

# a, b = detect_ridges(gray3, sigma=1.0)
# #cv.normalize(a,  a, 0, 255, cv.NORM_MINMAX)
# cv.imshow("ridges_max", a)
# #cv.normalize(b,  b, 0, 255, cv.NORM_MINMAX)
# cv.imshow("ridges_min", b)
#img2 = cv.imread('snapshot200.png') # trainImage

scaledown=5.0
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
resized_frame3 = cv.resize(img3, (0,0), fx=(scaledown*0.3), fy=(scaledown*0.3)) 
resized_frame4 = cv.resize(img4, (0,0), fx=(scaledown*0.3), fy=(scaledown*0.3)) 
comb = np.vstack((resized_frame3, resized_frame4))
cv.imwrite("zzz_comb.bmp", comb)
#plt.imshow(img3),plt.show()
#cv.namedWindow("result", cv.WND_PROP_FULLSCREEN)
#cv.setWindowProperty("result",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
cv.imshow("result", comb)


# Show the wrapped pic
matches = matches[:Howmany]
ptsA = []
ptsB = []
kps1 = np.float32([kp.pt for kp in kp1])
kps2 = np.float32([kp.pt for kp in kp2])
matches_p = []
for m in matches:
    matches_p.append((m.trainIdx, m.queryIdx))
    ptsA.append(kps1[m.queryIdx])
    ptsB.append(kps2[m.trainIdx])

# computing a homography requires at least 4 matches
# construct the two sets of points
# ptsA = np.float32([kp1[i] for (_, i) in matches_p])
# ptsB = np.float32([kp2[i] for (i, _) in matches_p])
 
# compute the homography between the two sets of points
(H, status) = cv.findHomography(np.asarray(ptsA), np.asarray(ptsB), cv.RANSAC)
result = cv.warpPerspective(resized_img1, H, (resized_img1.shape[1] + resized_img2.shape[1], resized_img1.shape[0]))

cv.imshow("wrap", result)

cv.waitKey(0)   
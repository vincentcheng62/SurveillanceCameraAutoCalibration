import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
#cap = cv.VideoCapture('onbridge.MOV')
#ret, img1 = cap.read()
#img1 = cv.imread('onbridge.bmp') # trainImage
#img2 = cv.imread('snapshot200_unity.png') # trainImage?
img1 = cv.imread('IMG_7094_half.JPG') # trainImage
#img2 = cv.imread('IMG_7094_half_-15.JPG') # trainImage?
img2 = cv.imread('screen_1920x1080_4.png') # trainImage?


scaledownh=2.0
imgsample = img1.copy()
imgsample = cv.resize(imgsample, (0,0), fx=(1/scaledownh), fy=(1/scaledownh)) 
gray = cv.cvtColor(imgsample,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(3,3),0)
#blur = cv.medianBlur(gray,3)
edges = cv.Canny(blur,150,300,apertureSize = 3)
ridge_filter = cv.ximgproc.RidgeDetectionFilter_create(ksize=3)
ridges = ridge_filter.getRidgeFilteredImage(blur)
im1, contours, hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#imgcon = imgsample.copy()
imgcon = np.zeros(edges.shape, dtype=np.uint8)
#cv.drawContours(imgcon, contours, -1, (0,255,0), 3)
cv.drawContours(imgcon, contours, -1, 255, 3)
cv.imshow("contour", imgcon)
cv.imwrite("contour.bmp", contour)
#cv.imshow("edge", edges)
# Calculate Moments
#moments = cv.moments(im)
# Calculate Hu Moments
#huMoments = cv.HuMoments(moments)

imgsample2 = img2.copy()
imgsample2 = cv.resize(imgsample2, (0,0), fx=(1/scaledownh), fy=(1/scaledownh)) 
gray2 = cv.cvtColor(imgsample2,cv.COLOR_BGR2GRAY)
blur2 = cv.GaussianBlur(gray2,(3,3),0)
#blur2 = cv.medianBlur(gray2,3)
edges2 = cv.Canny(blur2,150,300,apertureSize = 3)
#cv.imshow("edge", edges)
im2, contours2, hierarchy = cv.findContours(edges2,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#imgcon2 = imgsample2.copy()
imgcon2 = np.zeros(edges2.shape, dtype=np.uint8)
#cv.drawContours(imgcon2, contours2, -1, (0,255,0), 3)
cv.drawContours(imgcon2, contours2, -1, 255, 3)
cv.imshow("contour2", imgcon2)
cv.imwrite("contour2.bmp", contour2)

#ret = cv.matchShapes(contours[0],contours2[0],1,0.0)
ret = cv.matchShapes(imgcon,imgcon2,1,0.0)
print ("match shape score: ", ret)

#blur = cv.GaussianBlur(gray,(5,5),0)
# blur = cv.medianBlur(gray,7)
# th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,13,0)
# cv.imshow("adaptive_thd", th3)

minLineLength = 30
maxLineGap = 15
lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#imgsamplehlp = imgsample.copy()
for line in lines:
    cv.line(imgsample,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,255,0),2)
#cv.imshow("houghlineP", imgsamplehlp)
HoughThd=100

lines = cv.HoughLines(edges,1,3.0*np.pi/180,HoughThd)
for i in  range(0,lines.shape[0]):
    rho,theta = lines[i,0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(imgsample,(x1,y1),(x2,y2),(0,0,255),1)
#cv.imshow("hough", imgsample)

#Draw a mask
mask = np.zeros(edges.shape, dtype=np.uint8)
for i in  range(0,lines.shape[0]):
    rho,theta = lines[i,0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(mask,(x1,y1),(x2,y2),255,7)
#cv.imshow("mask", mask)


maskout = cv.bitwise_and(edges, edges, mask=mask)     
#cv.imshow("maskout", maskout)
#cv.imwrite("maskout.bmp", maskout)

lines2 = cv.HoughLines(edges2,1,3.0*np.pi/180,HoughThd)
for i in  range(0,lines2.shape[0]):
    rho,theta = lines2[i,0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(imgsample2,(x1,y1),(x2,y2),(0,0,255),1)

#Draw a mask
mask2 = np.zeros(edges2.shape, dtype=np.uint8)
for i in  range(0,lines2.shape[0]):
    rho,theta = lines2[i,0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(mask2,(x1,y1),(x2,y2),255,7)
#cv.imshow("mask2", mask2)


maskout2 = cv.bitwise_and(edges2, edges2, mask=mask2)     
#cv.imshow("maskout2", maskout2)
#cv.imwrite("maskout2.bmp", maskout2)

# cv.imshow("hough2", imgsample2)
# resized_frameh3 = cv.resize(imgsample, (0,0), fx=(scaledownh*0.3), fy=(scaledownh*0.3)) 
# resized_frameh4 = cv.resize(imgsample2, (0,0), fx=(scaledownh*0.3), fy=(scaledownh*0.3)) 
# combh = np.vstack((resized_frameh3, resized_frameh4))
# cv.imwrite("zzz_houghresult.bmp", combh)
# cv.imshow("houghresult", combh)


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
bf = cv.BFMatcher()

# Match descriptors.
#matches = bf.match(des1,des2)
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
ratioo=0.75
good_matches = []
for m,n in matches:
    if m.distance < ratioo*n.distance:
        good_matches.append(m)

# Sort them in the order of their distance.
good_matches = sorted(good_matches, key = lambda x:x.distance)

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
img3 = cv.drawMatches(resized_img1,kp1,resized_img2,kp2,good_matches[:Howmany],None, flags=2)
img4 = cv.drawMatches(resized_img1,kp1,resized_img2,kp2,good_matches[:Howmany],None, flags=0)
cv.imwrite("zzz3.bmp", img3)
cv.imwrite("zzz4.bmp", img4)
#cv.imwrite("onbridge.bmp", img1)
resized_frame3 = cv.resize(img3, (0,0), fx=(scaledown*0.3), fy=(scaledown*0.3)) 
resized_frame4 = cv.resize(img4, (0,0), fx=(scaledown*0.3), fy=(scaledown*0.3)) 
comb = np.vstack((resized_frame3, resized_frame4))
#cv.imwrite("zzz_comb.bmp", comb)
#plt.imshow(img3),plt.show()
#cv.namedWindow("result", cv.WND_PROP_FULLSCREEN)
#cv.setWindowProperty("result",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
#cv.imshow("result", comb)
cv.waitKey(0)   
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import imutils
#cap = cv.VideoCapture('onbridge.MOV')
#ret, img1 = cap.read()
#img1 = cv.imread('onbridge.bmp') # trainImage
#img2 = cv.imread('snapshot200_unity.png') # trainImage?
#img1 = cv.imread('IMG_7094_half.JPG') # trainImage

# img1 = cv.imread('IMG_7094_half.JPG') # trainImage
# img2 = cv.imread('IMG_7371_half.JPG') # trainImage?


#img2 = cv.imread('IMG_7094_half_-15.JPG') # trainImage?
#img2 = cv.imread('screen_1920x1080_4.png') # trainImage?
#img2 = cv.imread('DJI_0029_half.JPG') # trainImage?

# img1 = cv.imread('onbridge1.jpg') # trainImage?
# img2 = cv.imread('onbridge2.jpg') # trainImage?

# img1 = cv.imread('sift_testing/iphone_3pm/IMG_7661.JPG') # trainImage?
# img2 = cv.imread('sift_testing/iphone_10am/IMG_7755.JPG') # trainImage?

#img1 = cv.imread('sift_testing/lg_3pm/20190320_110602.jpg') # trainImage?
#img2 = cv.imread('sift_testing/lg_10am/20190322_091839.jpg') # trainImage?

#img1 = cv.imread('sift_testing/samsung_3pm/20180727_060317.jpg') # trainImage?
#img2 = cv.imread('sift_testing/samsung_10am/20180729_041921.jpg') # trainImage?

#img1 = cv.imread('sift_testing/iphone_3pm/IMG_7697.JPG') # trainImage?
#img2 = cv.imread('sift_testing/lg_3pm/20190320_110353.jpg') # trainImage?

#img1 = cv.imread('sift_testing/samsung_10am/20180729_042311.jpg') # trainImage?
#img2 = cv.imread('sift_testing/lg_10am/20190322_091336.jpg') # trainImage?

# img1 = cv.imread('sift_testing/samsung_3pm/20180727_060252.jpg') # trainImage?
# img2 = cv.imread('sift_testing/iphone_3pm/IMG_7670.JPG') # trainImage?

# compare same cam same time slight change in fov

#img1 = cv.imread('sift_testing/iphone_10am/IMG_7752.JPG') # trainImage?
#img2 = cv.imread('sift_testing/iphone_10am/IMG_7753.JPG') # trainImage?

#img1 = cv.imread('sift_testing/iphone_3pm/IMG_7667.JPG') # trainImage?
#img2 = cv.imread('sift_testing/iphone_3pm/IMG_7668.JPG') # trainImage?

#img1 = cv.imread('sift_testing/lg_10am/20190322_090547.jpg') # trainImage?
#img2 = cv.imread('sift_testing/lg_10am/20190322_090549.jpg') # trainImage?

#img1 = cv.imread('sift_testing/lg_3pm/20190320_110212.jpg') # trainImage?
#img2 = cv.imread('sift_testing/lg_3pm/20190320_110215.jpg') # trainImage?

#img1 = cv.imread('sift_testing/samsung_10am/20180729_042445.jpg') # trainImage?
#img2 = cv.imread('sift_testing/samsung_10am/20180729_042446.jpg') # trainImage?

# img1 = cv.imread('sift_testing/samsung_3pm/20180727_061434.jpg') # trainImage?
# img2 = cv.imread('sift_testing/samsung_3pm/20180727_061435.jpg') # trainImage?



# compare same cam same time large change in fov

#img1 = cv.imread('sift_testing/iphone_10am/IMG_7751.JPG') # trainImage?
#img2 = cv.imread('sift_testing/iphone_10am/IMG_7752.JPG') # trainImage?

#img1 = cv.imread('sift_testing/iphone_3pm/IMG_7667.JPG') # trainImage?
#img2 = cv.imread('sift_testing/iphone_3pm/IMG_7671.JPG') # trainImage?

#img1 = cv.imread('sift_testing/lg_10am/20190322_090547.jpg') # trainImage?
#img2 = cv.imread('sift_testing/lg_10am/20190322_090554.jpg') # trainImage?

#img1 = cv.imread('sift_testing/lg_3pm/20190320_110213.jpg') # trainImage?
#img2 = cv.imread('sift_testing/lg_3pm/20190320_110217.jpg') # trainImage?

#img1 = cv.imread('sift_testing/samsung_10am/20180729_042453.jpg') # trainImage?
#img2 = cv.imread('sift_testing/samsung_10am/20180729_042446.jpg') # trainImage?

# img1 = cv.imread('sift_testing/samsung_3pm/20180727_061435.jpg') # trainImage?
# img2 = cv.imread('sift_testing/samsung_3pm/20180727_061437.jpg') # trainImage?


# different lighting, camera and large change in fov
img1 = cv.imread('sift_testing/iphone_10am/IMG_7751.JPG') # trainImage?
img2 = cv.imread('sift_testing/lg_3pm/20190320_105515.jpg') # trainImage?

#img1 = cv.imread('sift_testing/lg_10am/20190322_090614.jpg') # trainImage?
#img2 = cv.imread('sift_testing/samsung_3pm/20180727_060314.jpg') # trainImage?

# img1 = cv.imread('sift_testing/samsung_10am/20180729_042311.jpg') # trainImage?
# img2 = cv.imread('sift_testing/iphone_10am/IMG_7776.JPG') # trainImage?

# img1 = cv.imread('sift_testing/iphone_10am/IMG_7781.JPG') # trainImage?
# img2 = cv.imread('sift_testing/lg_3pm/20190320_110359.jpg') # trainImage?

# img1 = cv.imread('sift_testing/lg_10am/20190322_091841.jpg') # trainImage?
# img2 = cv.imread('sift_testing/samsung_3pm/20180727_061430.jpg') # trainImage?



# after_rot = imutils.rotate_bound(img2, 10)
# img2 = after_rot.copy()

scaledownh=2.0
imgsample = img1.copy()
imgsample = cv.resize(imgsample, (0,0), fx=(1/scaledownh), fy=(1/scaledownh)) 
gray = cv.cvtColor(imgsample,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(3,3),0)
#blur = cv.medianBlur(gray,3)
edges = cv.Canny(blur,150,300,apertureSize = 3)
# ridge_filter = cv.ximgproc.RidgeDetectionFilter_create(ksize=3)
# ridges = ridge_filter.getRidgeFilteredImage(blur)
# im1, contours, hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# #imgcon = imgsample.copy()
imgcon = np.zeros(edges.shape, dtype=np.uint8)
# #cv.drawContours(imgcon, contours, -1, (0,255,0), 3)
# cv.drawContours(imgcon, contours, -1, 255, 3)
#cv.imshow("contour", imgcon)

cropmargin=50
cropcon=imgcon[cropmargin:imgcon.shape[0]-cropmargin, cropmargin:imgcon.shape[1]-cropmargin]
#cropcon = cv.resize(cropcon, (0,0), fx=(1/3.0), fy=(1/3.0)) 
#cv.imwrite("contour.bmp", cropcon)
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
# im2, contours2, hierarchy = cv.findContours(edges2,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# #imgcon2 = imgsample2.copy()
# imgcon2 = np.zeros(edges2.shape, dtype=np.uint8)
# #cv.drawContours(imgcon2, contours2, -1, (0,255,0), 3)
# cv.drawContours(imgcon2, contours2, -1, 255, 3)
#cv.imshow("contour2", imgcon2)
#imgcon2 = cv.resize(imgcon2, (0,0), fx=(1/3.0), fy=(1/3.0)) 
#cv.imwrite("contour2.bmp", imgcon2)

# #ret = cv.matchShapes(contours[0],contours2[0],1,0.0)
# ret = cv.matchShapes(imgcon,imgcon2,1,0.0)
# print ("match shape score: ", ret)

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

scaledown = 6.0
#resized_img1 = img1
#resized_img2 = img2
resized_img1 = cv.resize(img1, (0,0), fx=(1/scaledown), fy=(1/scaledown)) 
resized_img2 = cv.resize(img2, (0,0), fx=(1/scaledown), fy=(1/scaledown)) 

# scaledown=2.0
# resized_img1 = cv.resize(cropcon, (0,0), fx=(1/scaledown), fy=(1/scaledown)) 
# resized_img2 = cv.resize(imgcon2, (0,0), fx=(1/scaledown), fy=(1/scaledown)) 

# scaledown=3.0
# resized_img1 = cropcon
# resized_img2 = imgcon2

# Initiate SIFT detector
#detector = cv.ORB_create(nfeatures=2000, scaleFactor=1.1)
#detector = cv.BRISK_create(thresh=10, octaves=5, patternScale=1.1)

# SIFT para
# nfeatures	The number of best features to retain. The features are ranked by their scores 
# (measured in SIFT algorithm as the local contrast)
# nOctaveLayers	The number of layers in each octave. 3 is the value used in D. Lowe paper. 
# The number of octaves is computed automatically from the image resolution.
# contrastThreshold	The contrast threshold used to filter out weak features in semi-uniform 
# (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
# edgeThreshold	The threshold used to filter out edge-like features. Note that the its meaning 
# is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features 
# are filtered out (more features are retained).
# sigma	The sigma of the Gaussian applied to the input image at the octave #0. If your image 
# is captured with a weak camera with soft lenses, you might want to reduce the number.

#detector = cv.xfeatures2d.SIFT_create(nfeatures=500, contrastThreshold=0.05, edgeThreshold=3)

# pip3 install opencv-contrib-python==3.4.2.16
detector = cv.xfeatures2d.SIFT_create()
#detector = cv.xfeatures2d.SURF_create()
# find the keypoints and descriptors with SIFT

cropmargin=100
cropimg1=resized_img1[cropmargin:resized_img1.shape[0]-cropmargin, cropmargin:resized_img1.shape[1]-cropmargin]

resized_img1 = cropimg1.copy()

kp1 = detector.detect(resized_img1,None)
print("len(kp1): ", len(kp1))
kp2 = detector.detect(resized_img2,None)
print("len(kp2): ", len(kp2))

kp1, des1 = detector.compute(resized_img1, kp1)
kp2, des2 = detector.compute(resized_img2, kp2)

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
#match(query, train)
knnnum=20
matches = bf.knnMatch(des1,des2,k=knnnum)

# Apply ratio test
#matches = sorted(matches, key = lambda x:x[0].distance, reverse=True)
matches = sorted(matches, key = lambda x:x[0].distance)
ratioo=0.7 # the smaller the less match
max_TM_thd=0.1
block_size=20
good_matches = []
for match in matches:
    mm = np.mean([match[1].distance, match[2].distance, match[3].distance, match[4].distance])
    if match[0].distance < ratioo*mm:
        good_matches.append(match[0])

        # top_pt = kp2[match[0].trainIdx].pt
        # second_pt = kp2[match[1].trainIdx].pt

        # print("top_pt, x:", top_pt[0], ", y: ", top_pt[1])
        # print("second_pt, x:", second_pt[0], ", y: ", second_pt[1])

        # if top_pt[0] > block_size and top_pt[0] < resized_img2.shape[0]-block_size and top_pt[1]> block_size and top_pt[1] < resized_img2.shape[1]-block_size and \
        #     second_pt[0] > block_size and second_pt[0] < resized_img2.shape[0]-block_size and second_pt[1]> block_size and second_pt[1] < resized_img2.shape[1]-block_size:
        #     top_region = resized_img2[int(top_pt[0]-block_size):int(top_pt[0]+block_size), int(top_pt[1]-block_size):int(top_pt[1]+block_size)]
        #     second_region = resized_img2[int(second_pt[0]-block_size):int(second_pt[0]+block_size), int(second_pt[1]-block_size):int(second_pt[1]+block_size)]
        #     #res = cv.matchTemplate(top_region,second_region,cv.TM_SQDIFF)
        #     #min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        #     diff = cv.absdiff(top_region, second_region)
        #     sum = cv.sumElems(diff)
        #     score = sum[0]*1.0/(255.0*block_size*2*block_size*2)
        #     print("sum: ", sum[0], " , score: ", score)
        #     if score > max_TM_thd:
        #         good_matches.append(match[0])
        #         print("Good match!!!!!!!!!!!!!!!!!!!!!")

# Sort them in the order of their distance.
# good_matches = sorted(good_matches, key = lambda x:x.distance)

print("No of matches: ", len(matches))
print("No of good matches: ", len(good_matches))

# Draw first 10 matches.
#img1p = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
#img2p = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
Howmany=35

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


#cv.imwrite("zzz3.bmp", img3)
#cv.imwrite("zzz4.bmp", img4)
#cv.imwrite("onbridge.bmp", img1)
compensate_scale=0.7
resized_frame3 = cv.resize(img3, (0,0), fx=(scaledown*compensate_scale), fy=(scaledown*compensate_scale)) 
resized_frame4 = cv.resize(img4, (0,0), fx=(scaledown*compensate_scale), fy=(scaledown*compensate_scale)) 

comb = np.vstack((resized_frame3, resized_frame4))
cv.imwrite("matching_result.bmp", comb)
#plt.imshow(img3),plt.show()
#cv.namedWindow("result", cv.WND_PROP_FULLSCREEN)
#cv.setWindowProperty("result",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
#cv.imshow("result", comb)
#cv.waitKey(0)   


# Also plot the knn matches
# for j in range(0, 50):
#     img = cv.drawMatches(resized_img1,kp1,resized_img2,kp2,matches[j],None, flags=2)
#     resized_frame = cv.resize(img, (0,0), fx=(scaledown*compensate_scale*1.2), fy=(scaledown*compensate_scale*1.2)) 
#     cv.imwrite("match_of_" + str(j) + ".bmp", resized_frame)

#     #print all the distance
#     print("")
#     print("Feature ", j, ": ", end='')
#     for z in range(0, knnnum):
#         print(matches[j][z].distance, " ", end='')
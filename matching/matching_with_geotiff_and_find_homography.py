import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
import time, datetime
import math
from osgeo import gdal
import pyproj

print("cv2.__version__: ", cv.__version__)

###### Input parameters ######

IsWriteDownAllMatches=False
IsRatioTestOriginal=True
IsMappingDstPtToGPS=True
IsUsingMask=True

live_camera_img_path = 'marker6m.jpg'
live_camera_img_mask_path = 'outputmask.jpg'
# orthomosaic_img_path = 'marker.jpg'
orthomosaic_img_path = 'marker.jpg'
orthomosaic_img_mask_path = 'outputmask2.jpg'
orthomosaic_geotiff_path = 'marker.tif'

realimgscaledown=5.0
scaledown=8.0
compensate_scale=1.0 # the smaller, the smaller the size of "matching_result.bmp"
knnnum=5
ratioo=0.8 # ratio test, the lower the less matches
max_TM_thd=0.1 # threshold for template matching, not using now
block_size=20 # block size for local template matching, not using now
Howmany=100 # number of matches to visualize in the "matching_result.bmp", more will make the image looks chaos
ransacReprojThreshold = 0.5 # for homography fitting, the unit is meter

###### Input parameters ends ######

print("IsUsingMask: ", IsUsingMask)
print("IsWriteDownAllMatches: ", IsWriteDownAllMatches)
print("IsRatioTestOriginal: ", IsRatioTestOriginal)
print("IsMappingDstPtToGPS: ", IsMappingDstPtToGPS)
print("live_camera_img_path(img1): ", live_camera_img_path)
print("live_camera_img_mask_path(img1): ", live_camera_img_mask_path)
print("orthomosaic_img_path(img2): ", orthomosaic_img_path)
print("orthomosaic_img_mask_path(img2): ", orthomosaic_img_mask_path)
print("orthomosaic_geotiff_path: ", orthomosaic_geotiff_path)

start = time.time() 
img1 = cv.imread(live_camera_img_path) # live camera
img1_mask = cv.imread(live_camera_img_mask_path, cv.IMREAD_GRAYSCALE) # live camera
img2 = cv.imread(orthomosaic_img_path) # orthomosaic image
img2_mask = cv.imread(orthomosaic_img_mask_path, cv.IMREAD_GRAYSCALE) # orthomosaic image



print("original size(img1): ", img1.shape)
print("original size(img2): ", img2.shape)

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

output_dir = "./SaveIntermediateImageToDebug_"+str(st) +"/"
os.mkdir(output_dir)



print("scaledown factor for real image: ", realimgscaledown)
print("scaledown factor for orthomosaic(geotiff): ", scaledown)

resized_img1 = cv.resize(img1, (0,0), fx=(1/realimgscaledown), fy=(1/realimgscaledown)) 

#resized_img1 = img1.copy()
resized_img2 = cv.resize(img2, (0,0), fx=(1/scaledown), fy=(1/scaledown)) 

resized_img1_mask = None
resized_img2_mask = None

if IsUsingMask:
    resized_img1_mask = cv.resize(img1_mask, (0,0), fx=(1/realimgscaledown), fy=(1/realimgscaledown)) 
    resized_img2_mask = cv.resize(img2_mask, (0,0), fx=(1/scaledown), fy=(1/scaledown)) 

print("new size(img1): ", resized_img1.shape)
print("new size(img2): ", resized_img2.shape)

# scaledown=3.0
# resized_img1 = cropcon
# resized_img2 = imgcon2

# Initiate SIFT detector
#detector = cv.ORB_create(nfeatures=20000, scaleFactor=1.05)
#detector = cv.BRISK_create(thresh=10, octaves=8, patternScale=1.05)

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

#detector = cv.xfeatures2d.SIFT_create(nfeatures=50, contrastThreshold=0.05, edgeThreshold=3)
detector = cv.xfeatures2d.SIFT_create()

#detector = cv.xfeatures2d.SURF_create()
# find the keypoints and descriptors with SIFT

print("Start to detect keypoint...")
kp1 = detector.detect(resized_img1,resized_img1_mask)
kp2 = detector.detect(resized_img2,resized_img2_mask)

print("number of keypoint of img1: ", len(kp1))
print("number of keypoint of img2: ", len(kp2))

print("Start to compute descriptor...")
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

print("knn, k=: ", knnnum)
print("Start to knn matching...")
matches = bf.knnMatch(des1,des2,k=knnnum)

# Apply ratio test
# matches = sorted(matches, key = lambda x:x[0].distance)

good_matches = []

print("Start to ratio test to produce good_matches...")
for match in matches:
    if IsRatioTestOriginal is True:
        if match[0].distance < ratioo*match[1].distance:
            good_matches.append(match[0])
    else:
        if match[0].distance < ratioo*match[1].distance and match[0].distance < ratioo*match[2].distance and match[0].distance < ratioo*match[3].distance:
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

print("good_matches number: ", len(good_matches))

# Draw first 10 matches.
#img1p = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)import numpy as np

# Draw first 10 matches.
#img1p = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
#img2p = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)

print("start to find homography...")

print("ransacReprojThreshold in homography fitting: ", ransacReprojThreshold)
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

# print(src_pts)

# print("Scale back src_pt to real scale...")
# for pt in src_pts:
#     pt = [np.array([pt[0][0]*scaledown, pt[0][1]*scaledown])]

# print(src_pts)

dst_pts_gps=[]

if IsMappingDstPtToGPS:
    ds = gdal.Open(orthomosaic_geotiff_path)


    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    miny = gt[0]
    minx = gt[3] + width*gt[4] + height*gt[5] 
    maxy = gt[0] + width*gt[1] + height*gt[2]
    maxx = gt[3] 

    print("")
    print("Print out gps boundary of the ortho image:")
    print("minx(Latitude from south): ", minx)
    print("maxx(Latitude from north): ", maxx)
    print("miny(Longtitude from west): ", miny)
    print("maxy(Longtitude from east): ", maxy)

    print("original dst_pts: ", dst_pts)

    print("Map orthomosaic image px point to gps...")
    for pt in dst_pts:
        #print(pt[0][0], pt[0][1])

        # y coord in jpg => lat
        normalize_x = maxx+ (pt[0][1]/resized_img2.shape[0])*(minx-maxx)

        # x coord in jpg => long
        normalize_y = miny+ (pt[0][0]/resized_img2.shape[1])*(maxy-miny)

        new_pt = [np.array([normalize_x, normalize_y])]
        #print(new_pt)
        dst_pts_gps.append(new_pt)

    dst_pts=np.asarray(dst_pts_gps.copy())

    print("Mapped dst_pts: ", dst_pts)

M=None
mask=None
reprojectimgpts=None

if len(good_matches) < 4:
    print("Less than 4 good matches, cannot fit homography!")
else:
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,ransacReprojThreshold)
    print("Homography: ", M)
    print("mask: ", mask.transpose())

    if not M is None:
        reprojectimgpts = cv.perspectiveTransform(src_pts, M)







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
print("start rendering result....")

img3 = cv.drawMatches(resized_img1,kp1,resized_img2,kp2,good_matches[:Howmany],None, flags=2)
img4 = cv.drawMatches(resized_img1,kp1,resized_img2,kp2,good_matches[:Howmany],None, flags=0)


#cv.imwrite("zzz3.bmp", img3)
#cv.imwrite("zzz4.bmp", img4)
#cv.imwrite("onbridge.bmp", img1)

print("visualization compensation scale: ", compensate_scale)

resized_frame3 = cv.resize(img3, (0,0), fx=compensate_scale, fy=compensate_scale) 
resized_frame4 = cv.resize(img4, (0,0), fx=compensate_scale, fy=compensate_scale) 

comb = np.vstack((resized_frame3, resized_frame4))
cv.imwrite(os.path.join(output_dir, "matching_result.bmp"), comb)
#plt.imshow(img3),plt.show()
#cv.namedWindow("result", cv.WND_PROP_FULLSCREEN)
#cv.setWindowProperty("result",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
#cv.imshow("result", comb)
#cv.waitKey(0)   

print("How many pairs to visualize(at max): ", Howmany)
print("knn distance of the matches....")
# Also plot the knn matches
for j in range(0, Howmany):
    if IsWriteDownAllMatches:
        img = cv.drawMatches(resized_img1,kp1,resized_img2,kp2,matches[j],None, flags=2)
        resized_frame = cv.resize(img, (0,0), fx=(scaledown*compensate_scale*0.5), fy=(scaledown*compensate_scale*0.5)) 
        cv.imwrite(os.path.join(output_dir, "match_of_" + str(j) + ".bmp"), resized_frame)

    #print all the distance
    print("")
    print("Feature ", j, ": ", end='')
    for z in range(0, 5):
        print(matches[j][z].distance, " ", end='')


print("")
print("")
print("=====SUMMARY=====")
print("scaledown factor for orthomosaic(geotiff): ", scaledown)
print("knn, k=: ", knnnum)
print("original size(img1): ", img1.shape)
print("original size(img2): ", img2.shape)
print("new size(img1): ", resized_img1.shape)
print("new size(img2): ", resized_img2.shape)
print("number of keypoint of img1: ", len(kp1))
print("number of keypoint of img2: ", len(kp2))
print("matches number: ", len(matches))
print("good_matches number: ", len(good_matches))
print("ratio test threshold: ", ratioo)

if len(good_matches) > 3:
    print("Homography (from img1 px to geotiff gps coord): ", M)

    if not M is None:
        print("Point-wise reprojection error: ")
        totalfittingerror=0
        maskin=0
        _GEOD = pyproj.Geod(ellps='WGS84')
        for zz in range(len(src_pts)):
            if mask[zz] == 1:
                #error = math.sqrt(math.pow(reprojectimgpts[zz][0][0]-dst_pts[zz][0][0], 2)+math.pow(reprojectimgpts[zz][0][1]-dst_pts[zz][0][1], 2))
                #a,a2,d = _GEOD.inv(lon1,lat1,lon2,lat2) 
                #print(reprojectimgpts[zz][0][1],reprojectimgpts[zz][0][0],dst_pts[zz][0][1],dst_pts[zz][0][0])
                _,_, error = _GEOD.inv(reprojectimgpts[zz][0][1],reprojectimgpts[zz][0][0],dst_pts[zz][0][1],dst_pts[zz][0][0]) 
                print("src_pts index: ", zz, ", error(in m): ", error)
                totalfittingerror = totalfittingerror + error
                maskin=maskin+1
        fitting_error = totalfittingerror

        print("inliers number of homography: ", maskin, "/", len(good_matches))
        print("fittingerror in meter of the projected points: ", totalfittingerror/maskin)


        print("Storing the result in homography.yml...")
        fs_write = cv.FileStorage('homography.yml', cv.FILE_STORAGE_WRITE)
        fs_write.write("homography_matrix", M)
        fs_write.release()
    
    else:
        print("Homography is None, please check any points pair are duplicate or they lying on a straight line!!!")

else:
    print("Number of good matches <4, homography is not calculated!!!")

#img2p = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)

# print("start to find homography...")
# ransacReprojThreshold = 5.0
# src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
# dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

# # print(src_pts)

# # print("Scale back src_pt to real scale...")
# # for pt in src_pts:
# #     pt = [np.array([pt[0][0]*scaledown, pt[0][1]*scaledown])]

# # print(src_pts)

# dst_pts_gps=[]

# print("Map orthomosaic image px point to gps...")
# for pt in dst_pts:
#     normalize_x = minx+ (pt[0][0]/resized_img2.shape[1])*(maxx-minx)
#     normalize_y = miny+ (pt[0][1]/resized_img2.shape[0])*(maxy-miny)
#     new_pt = [np.array([normalize_x, normalize_y])]
#     dst_pts_gps.append(new_pt)

# dst_pts=np.asarray(dst_pts_gps.copy())

# M=None
# mask=None
# reprojectimgpts=None

# if len(good_matches) < 4:
#     print("Less than 4 good matches, cannot fit homography!")
# else:
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,ransacReprojThreshold)
#     reprojectimgpts = cv.perspectiveTransform(src_pts, M)




# Howmany=80


# # struct DrawMatchesFlags
# # {
# #     enum
# #     {
# #         DEFAULT = 0, // Output image matrix will be created (Mat::create),
# #                      // i.e. existing memory of output image may be reused.
# #                      // Two source images, matches, and single keypoints
# #                      // will be drawn.
# #                      // For each keypoint, only the center point will be
# #                      // drawn (without a circle around the keypoint with the
# #                      // keypoint size and orientation).
# #         DRAW_OVER_OUTIMG = 1, // Output image matrix will not be
# #                        // created (using Mat::create). Matches will be drawn
# #                        // on existing content of output image.
# #         NOT_DRAW_SINGLE_POINTS = 2, // Single keypoints will not be drawn.
# #         DRAW_RICH_KEYPOINTS = 4 // For each keypoint, the circle around
# #                        // keypoint with keypoint size and orientation will
# #                        // be drawn.
# #     };
# # };
# print("start rendering result....")

# img3 = cv.drawMatches(resized_img1,kp1,resized_img2,kp2,good_matches[:Howmany],None, flags=2)
# img4 = cv.drawMatches(resized_img1,kp1,resized_img2,kp2,good_matches[:Howmany],None, flags=0)


# #cv.imwrite("zzz3.bmp", img3)
# #cv.imwrite("zzz4.bmp", img4)
# #cv.imwrite("onbridge.bmp", img1)
# compensate_scale=0.3
# print("visualization compensation scale: ", compensate_scale)

# resized_frame3 = cv.resize(img3, (0,0), fx=(scaledown*compensate_scale), fy=(scaledown*compensate_scale)) 
# resized_frame4 = cv.resize(img4, (0,0), fx=(scaledown*compensate_scale), fy=(scaledown*compensate_scale)) 

# comb = np.vstack((resized_frame3, resized_frame4))
# cv.imwrite(os.path.join(output_dir, "matching_result.bmp"), comb)
# #plt.imshow(img3),plt.show()
# #cv.namedWindow("result", cv.WND_PROP_FULLSCREEN)
# #cv.setWindowProperty("result",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
# #cv.imshow("result", comb)
# #cv.waitKey(0)   

# print("How many pairs to visualize(at max): ", Howmany)
# print("knn distance of the matches (1st shortest distance to 5th shortest distance)....")
# # Also plot the knn matches
# for j in range(0, Howmany):
#     img = cv.drawMatches(resized_img1,kp1,resized_img2,kp2,matches[j],None, flags=2)
#     resized_frame = cv.resize(img, (0,0), fx=(scaledown*compensate_scale*0.5), fy=(scaledown*compensate_scale*0.5)) 
#     cv.imwrite(os.path.join(output_dir, "match_of_" + str(j) + ".bmp"), resized_frame)

#     #print all the distance
#     print("")
#     print("Feature ", j, ": ", end='')
#     for z in range(0, 5):
#         print(matches[j][z].distance, " ", end='')

# print("")
# print("")
# print("=====SUMMARY=====")
# print("scaledown factor for orthomosaic(geotiff): ", scaledown)
# print("knn, k=: ", knnnum)
# print("original size(img1): ", img1.shape)
# print("original size(img2): ", img2.shape)
# print("new size(img1): ", resized_img1.shape)
# print("new size(img2): ", resized_img2.shape)
# print("number of keypoint of img1: ", len(kp1))
# print("number of keypoint of img2: ", len(kp2))
# print("matches number: ", len(matches))
# print("good_matches number: ", len(good_matches))
# print("ratio test threshold: ", ratioo)

# if len(good_matches) > 3:
#     print("Homography (from img1 px to geotiff gps coord): ", M)
#     print("Point-wise error: ")
#     totalfittingerror=0
#     maskin=0
#     _GEOD = pyproj.Geod(ellps='WGS84')
#     for zz in range(len(src_pts)):
#         if mask[zz] == 1:
#             #error = math.sqrt(math.pow(reprojectimgpts[zz][0][0]-dst_pts[zz][0][0], 2)+math.pow(reprojectimgpts[zz][0][1]-dst_pts[zz][0][1], 2))
#             #a,a2,d = _GEOD.inv(lon1,lat1,lon2,lat2) 
#             #print(reprojectimgpts[zz][0][1],reprojectimgpts[zz][0][0],dst_pts[zz][0][1],dst_pts[zz][0][0])
#             _,_, error = _GEOD.inv(reprojectimgpts[zz][0][1],reprojectimgpts[zz][0][0],dst_pts[zz][0][1],dst_pts[zz][0][0]) 
#             print("error(in m): ", error)
#             totalfittingerror = totalfittingerror + error
#             maskin=maskin+1
#     fitting_error = totalfittingerror

#     print("inliers number of homography: ", maskin)
#     print("fittingerror in meter of the projected points: ", totalfittingerror/maskin)


#     print("Storing the result in homography.yml...")
#     fs_write = cv.FileStorage('homography.yml', cv.FILE_STORAGE_WRITE)
#     fs_write.write("homography_matrix", M)
#     fs_write.release()


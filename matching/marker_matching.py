import numpy as np 
import cv2 
import time, datetime
import os
import math
from osgeo import gdal
from sklearn.utils.linear_assignment_ import linear_assignment

def simplify_contour(contour, n_corners=4):
    '''
    Binary searches best `epsilon` value to force contour 
        approximation contain exactly `n_corners` points.

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    '''
    n_iter, max_iter = 0, 100
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour

        k = (lb + ub)/2.
        eps = k*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx

print("cv2.__version__: ", cv2.__version__)
   
###### Input parameters ######
print("########## Input parameters ##########")

MarkerSize=1.5 # in meter
orthomosaicscale=6.66  # mm/px, shown in photoscan UI
marker_area_variation=0.2  # 20%
area_to_perimeter_ratio_threshold=0.1 # 20%, for square it should be (+-0.25)

#img_path='real2.jpg'
#img_path='drone_ch.jpg'
img_path_list=['marker6m.jpg', 'marker.jpg']
#img_path='marker6m.jpg'

scaledownratio = [2.0, 3.0]

thrhd_grayvalue_high_for_whitepart_of_chessboard=170
thrhd_grayvalue_low_for_whitepart_of_chessboard=50
morph_open_kernel_radius_in_px=5
morph_close_kernel_radius_in_px=10
dilate_kernel_size_in_px_for_cb_mask_radius=6
dilate_kernel_size_for_white=60
approxPolyDP_epsilon=20

orthomosaic_geotiff_path = 'marker.tif'

print("orthomosaic_geotiff_path: ", orthomosaic_geotiff_path)
print("MarkerSize(in m): ", MarkerSize)
print("orthomosaicscale(mm/px): ", orthomosaicscale)
print("img_path_list: ", img_path_list)
print("marker_area_variation: ", marker_area_variation)
print("area_to_perimeter_ratio_threshold: ", area_to_perimeter_ratio_threshold)


print("thrhd_grayvalue_high_for_whitepart_of_chessboard: ", thrhd_grayvalue_high_for_whitepart_of_chessboard)
print("thrhd_grayvalue_low_for_whitepart_of_chessboard: ", thrhd_grayvalue_low_for_whitepart_of_chessboard)
print("morph_open_kernel_radius_in_px: ", morph_open_kernel_radius_in_px)
print("morph_close_kernel_radius_in_px: ", morph_close_kernel_radius_in_px)
print("dilate_kernel_size_in_px_for_cb_mask_radius: ", dilate_kernel_size_in_px_for_cb_mask_radius)
print("dilate_kernel_size_for_white: ", dilate_kernel_size_for_white)
print("approxPolyDP_epsilon: ", approxPolyDP_epsilon)
print("scaledownratio: ", scaledownratio)

colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_open_kernel_radius_in_px,morph_open_kernel_radius_in_px))   
morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_close_kernel_radius_in_px,morph_close_kernel_radius_in_px))   
dilate_kernel_for_cb_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate_kernel_size_in_px_for_cb_mask_radius, dilate_kernel_size_in_px_for_cb_mask_radius))    
dilate_kernel_white = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate_kernel_size_for_white, dilate_kernel_size_for_white))    



#Define common color HSV lower and upper
S_lower=100
S_Upper=255
V_lower=100
V_Upper=255
H_lower=-15
H_Upper=15

#From 0 to 180deg
redhsv=0 # dark red
red2hsv=180
yellowhsv=30
greenhsv=60
bluehsv=120
purplehsv=150

colorlist=[red2hsv, yellowhsv, greenhsv, bluehsv, purplehsv]

lowerhsvlist=[]
upperhsvlist=[]

for color in colorlist:
    lower = np.array([max(color+H_lower, 0),S_lower,V_lower])
    lowerhsvlist.append(lower)
    upper = np.array([min(color+H_Upper, 180),S_Upper,V_Upper])
    upperhsvlist.append(upper)


# Add white and black
#white, s close to 0, v close to 255
# lowerhsvlist.append(np.array([0,0,180]))
# upperhsvlist.append(np.array([180,50,255]))

# #black, V close to 0
# lowerhsvlist.append(np.array([0,0,0]))
# upperhsvlist.append(np.array([180,255,60]))

###### Input parameters ends ######

print("lowerhsvlist: ", lowerhsvlist)
print("upperhsvlist: ", upperhsvlist)

print("########## Input parameters ends ##########")
print(" ")

start = time.time() 

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

output_dir = "./PD_SaveIntermediateImageToDebug_"+str(st) +"/"
os.mkdir(output_dir)

#calculate the expected size of marker

side_in_px = (MarkerSize*1000)/(orthomosaicscale*scaledownratio[1])
print("[GeoTiff] expected marker side in px (in resized img): ", side_in_px)
area_in_px = side_in_px*side_in_px
print("[GeoTiff] expected marker area in px (in resized img): ", area_in_px)

result_contour_img=[]
result_contour_img.append([])
result_contour_img.append([])
result_contour_id=[]
result_contour_id.append([])
result_contour_id.append([])
result_contour_ROI=[]
result_contour_ROI.append([])
result_contour_ROI.append([])
result_contour_cnt=[]
result_contour_cnt.append([])
result_contour_cnt.append([])

frame_bwl_original_copy = [None] * 2
frame_bwh_original_copy = [None] * 2
frame_hsv_original_copy = [None] * 2
frame_img2 = [None] * 2

for i in range(0, len(img_path_list)):
    img_path = img_path_list[i]

    print("///////////////////////////////////////////////////")
    if i is 0:
        print("Doing survelliance camera round...")
    else:
        print("Doing orthomosaic image round...")

    # Reading image 
    img2 = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    print("input image original size: ", img2.shape)
    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_raw_color.jpg'), img2)

    img2 = cv2.resize(img2, (0,0), fx=(1/scaledownratio[i]), fy=(1/scaledownratio[i])) 
    print("input image resized size: ", img2.shape)

    frame_img2[i] = img2.copy()

    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_raw_color_resized.jpg'), img2)

    # convert to HSV

    imghsv = img2.copy()
    imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2HSV ) 
    frame_hsv_original_copy[i] = imghsv.copy()


    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_raw_hsv_resized.jpg'), imghsv)
    
    # Reading same image in another variable and  
    # converting to gray scale. 
    frame_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
    frame_gray = cv2.resize(frame_gray, (0,0), fx=(1/scaledownratio[i]), fy=(1/scaledownratio[i])) 

    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_raw_gray_resized.jpg'), frame_gray)    

    # set 255 to 0 since orthomosaic has white boundary
    for y in range(len(frame_gray)):
        for x in range(len(frame_gray[0])):
            if frame_gray[y][x] == 255:
                frame_gray[y][x] = 0


    hsvmask=None
    print("Start color segmentation...")
    for j in range(0, len(lowerhsvlist)):
        newhsvmask = cv2.inRange(imghsv, lowerhsvlist[j], upperhsvlist[j])
        cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_after' + str(j) + 'hsv.jpg'), newhsvmask)

        if hsvmask is None:
            hsvmask = newhsvmask.copy()
        else:
            hsvmask = cv2.bitwise_or(newhsvmask, hsvmask)   

    hsvmask = cv2.morphologyEx(hsvmask, cv2.MORPH_CLOSE, morph_open_kernel)
    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_1after_color_seg.jpg'), hsvmask)



    print("End color segmentation...")

    ##### old #####


    bw_ret,frame_bwh_original = cv2.threshold(frame_gray,thrhd_grayvalue_high_for_whitepart_of_chessboard,255,cv2.THRESH_BINARY)
    frame_bwh_original_copy[i] = frame_bwh_original.copy()
    frame_bwh = cv2.morphologyEx(frame_bwh_original, cv2.MORPH_OPEN, morph_open_kernel)
    frame_bwh = cv2.dilate(frame_bwh,dilate_kernel_for_cb_mask,iterations = 1)



    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_2threshold_w.jpg'), frame_bwh)


    frame_bwh_dilate = cv2.dilate(frame_bwh,dilate_kernel_white,iterations = 1)


    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_3dilate_w_as_mask.jpg'), frame_bwh_dilate)


    # set 255 to 0 since orthomosaic has white boundary
    for y in range(len(frame_gray)):
        for x in range(len(frame_gray[0])):
            if frame_gray[y][x] == 0:
                frame_gray[y][x] = 255

    bw_ret,frame_bwl_original = cv2.threshold(frame_gray,thrhd_grayvalue_low_for_whitepart_of_chessboard,255,cv2.THRESH_BINARY_INV)
    frame_bwl_original_copy[i] = frame_bwl_original.copy()
    frame_bwl = cv2.morphologyEx(frame_bwl_original, cv2.MORPH_CLOSE, morph_close_kernel)
    frame_bwl = cv2.dilate(frame_bwl,dilate_kernel_for_cb_mask,iterations = 1)

    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_4threshold_dark.jpg'), frame_bwl)

    frame_bw = cv2.bitwise_or(frame_bwh_original, frame_bwl, mask=frame_bwh_dilate)   

    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_5white_OR_dark.jpg'), frame_bw)


    ##### old ends #####

    color_or_bw = cv2.bitwise_or(frame_bw, hsvmask)  

    frame_bw = cv2.morphologyEx(frame_bw, cv2.MORPH_OPEN, morph_open_kernel)
    color_or_bw = cv2.dilate(color_or_bw,dilate_kernel_for_cb_mask,iterations = 1)
    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_6color_or_bw.jpg'), color_or_bw)

    # frame_bw = cv2.morphologyEx(frame_bw, cv2.MORPH_OPEN, morph_open_kernel)
    # frame_bw = cv2.dilate(frame_bw,dilate_kernel_for_cb_mask,iterations = 1)
    # frame_for_detect_cb = cv2.bitwise_and(frame_gray, frame_gray, mask=frame_bw)   

    # cv2.imwrite('parallelogram_frame_for_detect_cb.jpg', frame_for_detect_cb)

    # # Converting image to a binary image  
    # # (black and white only image). 
    # threshold = cv2.Canny(frame_for_detect_cb,100,200)
    # #_,threshold = cv2.threshold(img, 110, 255,  cv2.THRESH_BINARY) 

    # cv2.imwrite('parallelogram_canny.jpg', threshold)



    
    # Detecting shapes in image by selecting region  
    # with same colors or intensity. 
    print("Start finding contour....")
    image, contours, hierarchy =cv2.findContours(color_or_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for cnt in contours : 
        cv2.fillPoly(color_or_bw, pts =[cnt], color=(255,255,255))

    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_7fillPoly.jpg'), color_or_bw)



    # find contour again after filling polygon
    image, contours, hierarchy =cv2.findContours(color_or_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    frame_bw_color = cv2.cvtColor(color_or_bw, cv2.COLOR_GRAY2BGR)

    obj_id=0
    for cnt in contours : 
        color = colors[int(obj_id) % len(colors)]
        cv2.drawContours(frame_bw_color, cnt, -1, color, 5)
        obj_id=obj_id+1

    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_8findContours.jpg'), frame_bw_color)
    
    # Searching through every region selected to  
    # find the required polygon. 

    obj_id=0
    good_contours=[]
    approximg= img2.copy()
    hull = []

    area_lower_threshold=1000
    area_upper_threshold=30000
    # area_to_perimeter_ratio_lower_threshold=0.15
    # area_to_perimeter_ratio_upper_threshold=0.4
    area_to_perimeter_ratio_lower_threshold = 0.25*(1-area_to_perimeter_ratio_threshold*2.5)
    area_to_perimeter_ratio_upper_threshold = 0.25*(1+area_to_perimeter_ratio_threshold*2.5)

    if i is 1:
        area_lower_threshold = area_in_px*(1-marker_area_variation)
        area_upper_threshold = area_in_px*(1+marker_area_variation)
        area_to_perimeter_ratio_lower_threshold = 0.25*(1-area_to_perimeter_ratio_threshold)
        area_to_perimeter_ratio_upper_threshold = 0.25*(1+area_to_perimeter_ratio_threshold)

    print("Area lower threshold: ", area_lower_threshold)
    print("Area upper threshold: ", area_upper_threshold)
    print("area_to_perimeter_ratio lower threshold: ", area_to_perimeter_ratio_lower_threshold)
    print("area_to_perimeter_ratio upper threshold: ", area_to_perimeter_ratio_upper_threshold)

    print("Number of contour: ", len(contours))

    # fine tune approxPolyDP_epsilon to avoid making square to triangle
    # when square to triangle, 2r=>1.414r, so total decrease in perimeter = 0.59r
    max_epsilon = side_in_px*0.59*0.7
    print("max_epsilon: ", max_epsilon)

    approxPolyDP_epsilon = min(approxPolyDP_epsilon, max_epsilon)

    for cnt in contours : 
        color = colors[int(obj_id) % len(colors)]
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # Shortlisting the regions based on there area. 
        #Isconvex = cv2.isContourConvex(cnt)
        if area > area_lower_threshold and area < area_upper_threshold:  

            # hull.append(cv2.convexHull(cnt, False))   
            # cv2.drawContours(approximg, hull, 0, color, 5)      
            approx = cv2.approxPolyDP(cnt,  approxPolyDP_epsilon, False) 
            

            # print("len(approx): ", len(approx))

            # count=1
            # while not len(approx) is 3:
            #     if len(approx)>3:
            #         approx = cv2.approxPolyDP(cnt,  approxPolyDP_epsilon*(1+count*0.1), False) 
            #     elif len(approx)<3:
            #         approx = cv2.approxPolyDP(cnt,  approxPolyDP_epsilon*(1-count*0.1), False) 
            #         if count >= 10:
            #             break

            #     count=count+1
            #     print("count: ", count, " len(approx): ", len(approx))


            # print("approx: ", approx)


            if len(approx) > 2 and len(approx) < 7:
                approx = simplify_contour(cnt, 4)
                perimeter = cv2.arcLength(approx,True)
                ratio = math.sqrt(area)/(perimeter+1e-10)                
                
                #if ratio > area_to_perimeter_ratio_lower_threshold and ratio < area_to_perimeter_ratio_upper_threshold:
                print("obj_id: ", obj_id, " center: (", cX, ", ", cY, ") area: ", area, " perimeter: ", perimeter, "len(approx):", len(approx), " ratio: ", ratio)            
                cv2.drawContours(approximg, [approx], 0, color, 5) 
                good_contours.append(approx)

                x,y,w,h = cv2.boundingRect(approx)
                imgg = frame_gray[y:y+h,x:x+w]
                cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_contouridx_' + str(obj_id) + '_center_' + str(cX) + '_' + str(cY) + '_img.jpg'), imgg)
                imgg_color = img2[y:y+h,x:x+w]            
                cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_contouridx_' + str(obj_id) + '_center_' + str(cX) + '_' + str(cY) + '_img_color.jpg'), imgg_color)
                result_contour_img[i].append(imgg)
                result_contour_id[i].append(obj_id)
                result_contour_ROI[i].append((x,y,w,h))
                result_contour_cnt[i].append(approx)

        obj_id=obj_id+1

    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_9approxPolyDP.jpg'), approximg)

    outputmask = np.zeros(color_or_bw.shape, dtype="uint8") 
    for cnt in good_contours : 
        cv2.drawContours(outputmask, [cnt], 0, (255,255,255), -1) 

    outputmask = cv2.dilate(outputmask,dilate_kernel_for_cb_mask,iterations = 2)
    outputmask = cv2.resize(outputmask, (0,0), fx=scaledownratio[i], fy=scaledownratio[i])
    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_outputmask.jpg'), outputmask)

    raw = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    outputmask = cv2.cvtColor(outputmask, cv2.COLOR_GRAY2BGR ) 
    raw_masked = raw*outputmask
    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_raw_mask.jpg'), raw_masked)


    obj_id=0
    for cnt in good_contours : 
        color = colors[int(obj_id) % len(colors)]
    
        #print("obj_id: ", obj_id, ", approxcnt: ", cnt)
        rect = cv2.minAreaRect(cnt) 
        box = cv2.boxPoints(rect) 
        box = np.int0(box)
        cv2.drawContours(img2, [box], 0, color, 5)

        obj_id=obj_id+1

    # Showing the image along with outlined arrow. 
    #cv2.imshow('image2', img2)  
    cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_parallelogram_10detected.jpg'), img2)


print("=====================================================================")
print("Matching stage")

print("Method1: cv2.matchShapes")

dist_matrix = np.zeros((len(result_contour_img[0]),len(result_contour_img[1])),dtype=np.float32)
for d,img1 in enumerate(result_contour_img[0]):
    for t,img2 in enumerate(result_contour_img[1]):
        # with itself, score=0
        dist_matrix[d,t] = cv2.matchShapes(img1,img2,cv2.CONTOURS_MATCH_I1,0)

print("dist_matrix: ", dist_matrix)
matched_indices = linear_assignment(dist_matrix)
print("matched_indices: ", matched_indices)

for matches in matched_indices:
    print(result_contour_id[0][matches[0]], result_contour_id[1][matches[1]])


print("Method2: Area of hue vector")
huevector = [None] * 2
huevector[0] = [None] * len(result_contour_id[0])
huevector[1] = [None] * len(result_contour_id[1])

for qq in range(0, len(huevector)):
    for bb in range(0, len(result_contour_ROI[qq])):
        area_of_ROI = result_contour_ROI[qq][bb][2]*result_contour_ROI[qq][bb][3]
        print("area_of_ROI: ", area_of_ROI)
        area_in_hue = [0.0] * (len(lowerhsvlist)+2)
        x,y,w,h = result_contour_ROI[qq][bb]
        print("ROI: ", x, ", ", y, ", ", w, ", ", h)

        #color
        for yy in range(0, len(lowerhsvlist)):
            hsvmask = cv2.inRange(frame_hsv_original_copy[qq][y:y+h,x:x+w], lowerhsvlist[yy], upperhsvlist[yy])
            #hsvmask = cv2.morphologyEx(hsvmask, cv2.MORPH_OPEN, morph_open_kernel)
            area_of_non_zero = cv2.countNonZero(hsvmask)
            print("qq: ", qq, " idx: ", result_contour_id[qq][bb], ", area_of_non_zero: ", area_of_non_zero)
            #area_in_hue[yy] = 1.0*area_of_non_zero/area_of_ROI
            area_in_hue[yy] = 1.0*area_of_non_zero

        #white and black
        area_of_non_zero = cv2.countNonZero(frame_bwh_original_copy[qq][y:y+h,x:x+w])
        area_in_hue[len(lowerhsvlist)] = area_of_non_zero
        print("qq: ", qq, " idx: ", result_contour_id[qq][bb], ", area_of_non_zero: ", area_of_non_zero)

        area_of_non_zero = cv2.countNonZero(frame_bwl_original_copy[qq][y:y+h,x:x+w])
        area_in_hue[len(lowerhsvlist)+1] = area_of_non_zero
        print("qq: ", qq, " idx: ", result_contour_id[qq][bb], ", area_of_non_zero: ", area_of_non_zero)

        dst = np.asarray([0.0] * len(area_in_hue))
        cv2.normalize(np.asarray(area_in_hue), dst, 1.0, 0.0, cv2.NORM_L2)
        area_in_hue = dst
        
        huevector[qq][bb] = np.asarray(area_in_hue)
        print("area_in_hue: ", area_in_hue)


print("huevector: ", huevector)

dist_matrix = np.zeros((len(result_contour_img[0]),len(result_contour_img[1])),dtype=np.float32)
for d,img1 in enumerate(result_contour_img[0]):
    for t,img2 in enumerate(result_contour_img[1]):
        # with itself, score=0
        dist_matrix[d,t] = cv2.norm(huevector[0][d], huevector[1][t])

print("dist_matrix: ", dist_matrix)
matched_indices = linear_assignment(dist_matrix)
print("matched_indices: ", matched_indices)

for matches in matched_indices:
    print(result_contour_id[0][matches[0]], result_contour_id[1][matches[1]])


# Do the homography between the contour points (px to px only)

delta_idx=[0]*len(matched_indices) # only 4 combinations for 4 corners points
ransacReprojThreshold=5

src_pts=[]
dst_pts=[]
# prepare a list of center of mass of all contours as testing points
for matches in matched_indices:
    M = cv2.moments(result_contour_cnt[0][matches[0]])
    M1 = cv2.moments(result_contour_cnt[1][matches[1]])
    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]
    cX1 = M1["m10"] / M1["m00"]
    cY1 = M1["m01"] / M1["m00"]    
    new_pt = [np.array([cX, cY])]
    new_pt1 = [np.array([cX1, cY1])]
    src_pts.append(new_pt)
    dst_pts.append(new_pt1)

src_pts=np.asarray(src_pts.copy())     
dst_pts=np.asarray(dst_pts.copy())   

final_src_pts=None
final_dst_pts=None


counttt=0
for matches in matched_indices:
    bestidx=0
    besterror=1e9
    copylist= result_contour_cnt[1][matches[1]]
    

    # calculate the reprojection error of the center of mass, which has a unique answer
    for roundj in range(0, 4):

        if roundj is not 0:
            endvalue = copylist[-1]
            copylist = np.insert(copylist[:-1], 0, endvalue, axis=0)

        print("copylist: ", copylist)

        homography, mask = cv2.findHomography(result_contour_cnt[0][matches[0]], copylist, cv2.RANSAC,ransacReprojThreshold)

   
        reprojectimgpts = cv2.perspectiveTransform(src_pts, homography)
        reprojectionerror=0

        for zz in range(len(src_pts)):
            reprojectionerror = reprojectionerror + math.sqrt(math.pow(reprojectimgpts[zz][0][0]-dst_pts[zz][0][0], 2)+math.pow(reprojectimgpts[zz][0][1]-dst_pts[zz][0][1], 2))

        print("reprojectionerror: ", reprojectionerror)
        if reprojectionerror < besterror:
            besterror = reprojectionerror
            bestidx = roundj


    delta_idx[counttt] = bestidx
    counttt=counttt+1

    if final_src_pts is None:
        final_src_pts = np.asarray(result_contour_cnt[0][matches[0]])
    else:
        final_src_pts = np.append(final_src_pts, result_contour_cnt[0][matches[0]], axis=0)


    if final_dst_pts is None:
        final_dst_pts = np.asarray(copylist.copy())
    else:
        final_dst_pts = np.append(final_dst_pts, copylist, axis=0)

print("delta_idx: ", delta_idx)
print("final_dst_pts: ", final_dst_pts)

# Print out resized rgb image with the point index to see if the matching is correct
displayimg2 = frame_img2[0].copy()
for jjj in range(0, len(final_src_pts)):
    cv2.putText(displayimg2,str(jjj), (int(final_src_pts[jjj][0][0]) ,int(final_src_pts[jjj][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
cv2.imwrite(os.path.join(output_dir, 'round' + str(0) + '_parallelogram_raw_color_matched_index.jpg'), displayimg2)

displayimg2 = frame_img2[1].copy()
for jjj in range(0, len(final_dst_pts)):
    cv2.putText(displayimg2,str(jjj), (int(final_dst_pts[jjj][0][0]), int(final_dst_pts[jjj][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
cv2.imwrite(os.path.join(output_dir, 'round' + str(1) + '_parallelogram_raw_color_matched_index.jpg'), displayimg2)

# Load the geotiff
print("Load the geotiff to build the homography from survelliance camera to real gps coord")
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

print("Map orthomosaic image px point to gps...")
final_dst_pts_gps=[]
for pt in final_dst_pts:
    #print(pt[0][0], pt[0][1])

    # y coord in jpg => lat
    normalize_x = maxx+ (pt[0][1]/frame_img2[1].shape[0])*(minx-maxx)

    # x coord in jpg => long
    normalize_y = miny+ (pt[0][0]/frame_img2[1].shape[1])*(maxy-miny)

    new_pt = [np.array([normalize_x, normalize_y])]
    #print(new_pt)
    final_dst_pts_gps.append(new_pt)

final_dst_pts_gps=np.asarray(final_dst_pts_gps.copy())

homography, mask = cv2.findHomography(final_src_pts*scaledownratio[0], final_dst_pts_gps, cv2.RANSAC,1e-5)

print("mask: ", mask)
print("homography: ", homography)
print("Storing the result in homography.yml...")
fs_write = cv2.FileStorage('homography.yml', cv2.FILE_STORAGE_WRITE)
fs_write.write("homography_matrix", homography)
fs_write.release()

print('Whole process took seconds: ',time.time() - start)
   
# Exiting the window if 'q' is pressed on the keyboard. 
if cv2.waitKey(0) & 0xFF == ord('q'):  
    cv2.destroyAllWindows() 
import numpy as np 
import cv2 
import time, datetime
import os
import math
from osgeo import gdal
from sklearn.utils.linear_assignment_ import linear_assignment
import random

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


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def linelength(line):
    return math.sqrt(math.pow((line[2]-line[0]),2)+math.pow((line[3]-line[1]),2))

def slope_in_angle(line):
    return math.degrees(math.atan2(1.0*line[3]-1.0*line[1], (1.0*line[2]-1.0*line[0])))

def slope(line):
    return (1.0*line[3]-1.0*line[1])/(1.0*line[2]-1.0*line[0])

def checksameline(line1, line2):
    dist1 = math.sqrt(math.pow((line1[0]-line2[0]),2)+math.pow((line1[1]-line2[1]),2))
    dist2 = math.sqrt(math.pow((line1[0]-line2[2]),2)+math.pow((line1[1]-line2[3]),2))
    dist3 = math.sqrt(math.pow((line1[2]-line2[0]),2)+math.pow((line1[3]-line2[1]),2))
    dist4 = math.sqrt(math.pow((line1[2]-line2[2]),2)+math.pow((line1[3]-line2[3]),2))

    threshold = 2
    if min(dist1, dist2) < threshold and min(dist3, dist4) < threshold:
        return True
    else:
        return False

def intersect(line1, line2): 
    pt1 = (line1[0], line1[1])
    pt2 = (line1[2], line1[3])
    ptA = (line2[0], line2[1])
    ptB = (line2[2], line2[3])
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
        
        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;   x2, y2 = pt2
    dx1 = x2 - x1;  dy1 = y2 - y1

    dx1 = dx1 + random.random()*0.001
    dy1 = dy1 + random.random()*0.001

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;   xB, yB = ptB;
    dx = xB - x;  dy = yB - y;

    dx = dx + random.random()*0.001
    dy = dy + random.random()*0.001
    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)

    # now, the determinant should be OK
    DETinv = 1.0/DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))

    # return the average of the two descriptions
    xi = (x1 + r*dx1 + x + s*dx)/2.0
    yi = (y1 + r*dy1 + y + s*dy)/2.0
    return ( xi, yi, 1, r, s )

#input color image
def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            img, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)


def lines_merging(lines):
    new_extended_lines=[]
    lines_with_info=[]
    line_min_length=5
    lines_angle_diff_in_degree=5

    # make larger, something may miss the one and connect between (1) (-) (3)
    extension_ratio=2.0
    maxgap=50

    for line in lines:
        length=linelength(line[0])

        if length>line_min_length:
            lines_with_info.append([line[0], slope_in_angle(line[0]), length])

    for line1 in lines_with_info:

        CanMerge=False
        for line2 in lines_with_info:

            if not line1 is line2:

                abs_diff_in_angle = math.fabs(line1[1]-line2[1])
                if abs_diff_in_angle < lines_angle_diff_in_degree:

                    (xi, yi, valid, r, s) = intersect(line1[0], line2[0])
                    if valid == 1:

                        max_x = max(line1[0][0], line1[0][2], line2[0][0], line2[0][2])
                        min_x = min(line1[0][0], line1[0][2], line2[0][0], line2[0][2])
                        max_y = max(line1[0][1], line1[0][3], line2[0][1], line2[0][3])
                        min_y = min(line1[0][1], line1[0][3], line2[0][1], line2[0][3])
                        # r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
                        # s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)

                        if xi >= min_x and xi <= max_x and yi >= min_y and yi <= max_y:

                            if r > 0: # intersection is along line1 original direction
                                if r >=1 and r < 1.0+extension_ratio and (r-1)*line1[2]< maxgap:
                                    if s > 0 :
                                        if s >=1 and s < 1.0+extension_ratio and (s-1)*line2[2]< maxgap:                                
                                            new_extended_lines.append([[line1[0][0], line1[0][1], xi, yi]])
                                            new_extended_lines.append([[line2[0][0], line2[0][1], xi, yi]])
                                            CanMerge=True

                                    else:
                                        if s >= -1*extension_ratio and (-s)*line2[2]< maxgap:
                                            new_extended_lines.append([[line1[0][0], line1[0][1], xi, yi]])
                                            new_extended_lines.append([[line2[0][2], line2[0][3], xi, yi]])   
                                            CanMerge=True                                 
                            
                            else:
                                if r >= -1*extension_ratio and (-r)*line1[2]< maxgap:
                                    if s > 0 :
                                        if s >=1 and s < 1.0+extension_ratio and (s-1)*line2[2]< maxgap:                                
                                            new_extended_lines.append([[line1[0][2], line1[0][3], xi, yi]])
                                            new_extended_lines.append([[line2[0][0], line2[0][1], xi, yi]])
                                            CanMerge=True

                                    else:
                                        if s >= -1*extension_ratio and (-s)*line2[2]< maxgap:
                                            new_extended_lines.append([[line1[0][2], line1[0][3], xi, yi]])
                                            new_extended_lines.append([[line2[0][2], line2[0][3], xi, yi]]) 
                                            CanMerge=True   

                    else:
                        print("Invalid lines")
                # else:
                #     print("Invalid angle: ", abs_diff_in_angle)

        # also append the original segment if no merging happen to him
        if not CanMerge:
            new_extended_lines.append(line1)

    return new_extended_lines


print("cv2.__version__: ", cv2.__version__)
   
###### Input parameters ######
print("########## Input parameters ##########")

MarkerSize=1.5 # in meter
orthomosaicscale=12.4 #6.66  # mm/px, shown in photoscan UI
marker_area_variation=0.22  # 20%
area_to_perimeter_ratio_threshold=0.12 # 20%, for square it should be (+-0.25)

#img_path='real2.jpg'
#img_path='drone_ch.jpg'
#img_path_list=['marker6m.jpg', 'ted2.jpg']
img_path_list=['DJI_0161.JPG', 'marker40m.jpg']
orthomosaic_geotiff_path = 'marker40m.tif'
#img_path='marker6m.jpg'

# dont scale down survelliance camera image, since marker at the far end will be very small
scaledownratio = [1.0, 1.0]

# 2 are different since in general ortho geotiff photos are brighter
thrhd_grayvalue_high_for_whitepart_of_chessboard=[175,220]
thrhd_grayvalue_low_for_whitepart_of_chessboard=[50,70]
morph_open_kernel_radius_in_px=5
morph_close_kernel_radius_in_px=10
dilate_kernel_size_in_px_for_cb_mask_radius=6
dilate_kernel_size_for_white=60
approxPolyDP_epsilon=20
IsFindDarkAlsoNearSegmentedWhite=False
IsFindWhiteOnlyNearColorSeg=False
IsUsingRefinedCorner=True
RefinedCornerThreshold=[18,20]

length_approx_threshold=[2,7]
area_lower_threshold=1000
area_upper_threshold=15000
boundingrectmargin=[10, 20]
mergedlineminthreshold=[12, 45]
reflective_surface_thd=240

print("orthomosaic_geotiff_path: ", orthomosaic_geotiff_path)
print("MarkerSize(in m): ", MarkerSize)
print("orthomosaicscale(mm/px): ", orthomosaicscale)
print("img_path_list: ", img_path_list)
print("marker_area_variation: ", marker_area_variation)
print("area_to_perimeter_ratio_threshold: ", area_to_perimeter_ratio_threshold)
print("IsUsingRefinedCorner: ", IsUsingRefinedCorner)
print("IsFindDarkAlsoNearSegmentedWhite: ", IsFindDarkAlsoNearSegmentedWhite)
print("IsFindWhiteOnlyNearColorSeg: ", IsFindWhiteOnlyNearColorSeg)
print("length_approx_threshold: ", length_approx_threshold)
print("boundingrectmargin: ", boundingrectmargin)
print("reflective_surface_thd: ", reflective_surface_thd)
print("RefinedCornerThreshold: ", RefinedCornerThreshold)


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
yellowhsv=42
greenhsv=80
tealhsv=105
bluehsv=120
purplehsv=130

colorlist=[red2hsv, yellowhsv, greenhsv, tealhsv, bluehsv, purplehsv]

lowerhsvlist=[]
upperhsvlist=[]

for color in colorlist:
    lower = np.array([max(color+H_lower, 0),S_lower,V_lower])
    lowerhsvlist.append(lower)
    upper = np.array([min(color+H_Upper, 180),S_Upper,V_Upper])
    upperhsvlist.append(upper)

# special treatment for our yellow, s is lower to 30%
# 255*0.3 = 76.5
# 255*0.9 = 229.5
lowerhsvlist[1][1]=55
lowerhsvlist[1][2]=140

# special treatment for our green, S=66%, V=80%
lowerhsvlist[2][1]=150
lowerhsvlist[2][2]=130

# special treatment for teal, S=60%, V=30%
lowerhsvlist[3][1]=130
upperhsvlist[3][1]=170
lowerhsvlist[3][2]=60
upperhsvlist[3][2]=90


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

output_dir = "./PD_Debug_"+str(st) +"/"
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
    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_1raw_color.jpg'), img2)

    img2 = cv2.resize(img2, (0,0), fx=(1/scaledownratio[i]), fy=(1/scaledownratio[i])) 
    print("input image resized size: ", img2.shape)

    frame_img2[i] = img2.copy()

    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_2raw_color_resized.jpg'), img2)

    # convert to HSV

    imghsv = img2.copy()
    imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2HSV ) 
    print("Convert image to HSV...")
    frame_hsv_original_copy[i] = imghsv.copy()


    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_3raw_hsv_resized.jpg'), imghsv)
    
    # Reading same image in another variable and  
    # converting to gray scale. 

    print("Read to gray image again...")    
    frame_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 

    print("Resize to gray image")       
    frame_gray = cv2.resize(frame_gray, (0,0), fx=(1/scaledownratio[i]), fy=(1/scaledownratio[i])) 

    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_4raw_gray_resized.jpg'), frame_gray)    


    print("Set highly reflective surface as 0...")
    # set 255 to 0 since orthomosaic has white boundary
    frame_gray[frame_gray > reflective_surface_thd] = 0

    # for y in range(len(frame_gray)):
    #     for x in range(len(frame_gray[0])):
    #         if frame_gray[y][x] > reflective_surface_thd: # since marker white is not reflective surface, will not be very close to 255
    #             frame_gray[y][x] = 0


    hsvmask=None
    print("Start color segmentation...")
    for j in range(0, len(lowerhsvlist)):
        newhsvmask = cv2.inRange(imghsv, lowerhsvlist[j], upperhsvlist[j])
        cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_5after' + str(j) + 'hsv_in_' + str(namestr(colorlist[j], globals())) + '.jpg'), newhsvmask)

        if hsvmask is None:
            hsvmask = newhsvmask.copy()
        else:
            hsvmask = cv2.bitwise_or(newhsvmask, hsvmask)   

    hsvmask = cv2.morphologyEx(hsvmask, cv2.MORPH_CLOSE, morph_open_kernel)
    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_6after_color_seg.jpg'), hsvmask)



    print("End color segmentation...")

    ##### old #####


    bw_ret,frame_bwh_original = cv2.threshold(frame_gray,thrhd_grayvalue_high_for_whitepart_of_chessboard[i],255,cv2.THRESH_BINARY)

    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_7threshold_white_original.jpg'), frame_bwh_original)    

    frame_bw=frame_bwh_original.copy()
    frame_bwh_original_copy[i] = frame_bwh_original.copy()

    bw_ret,frame_bwl_original = cv2.threshold(frame_gray,thrhd_grayvalue_low_for_whitepart_of_chessboard[i],255,cv2.THRESH_BINARY_INV)
    frame_bwl_original_copy[i] = frame_bwl_original.copy()    

    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_8threshold_dark_original.jpg'), frame_bwl_original)    

    if IsFindDarkAlsoNearSegmentedWhite:
        
        frame_bwh = cv2.morphologyEx(frame_bwh_original, cv2.MORPH_OPEN, morph_open_kernel)
        frame_bwh = cv2.dilate(frame_bwh,dilate_kernel_for_cb_mask,iterations = 1)

        frame_bwh_dilate = cv2.dilate(frame_bwh,dilate_kernel_white,iterations = 1)


        cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_9dilate_w_as_mask.jpg'), frame_bwh_dilate)


        # set 255 to 0 since orthomosaic has white boundary
        for y in range(len(frame_gray)):
            for x in range(len(frame_gray[0])):
                if frame_gray[y][x] == 0:
                    frame_gray[y][x] = 255


        frame_bwl = cv2.morphologyEx(frame_bwl_original, cv2.MORPH_CLOSE, morph_close_kernel)
        frame_bwl = cv2.dilate(frame_bwl,dilate_kernel_for_cb_mask,iterations = 1)



        frame_bw = cv2.bitwise_or(frame_bwh_original, frame_bwl, mask=frame_bwh_dilate)   

        cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_10white_OR_dark.jpg'), frame_bw)
        print("Merging white and dark segmentation...")


    ##### old ends #####
    color_or_bw=None
    if IsFindWhiteOnlyNearColorSeg:
        hsvmask_asmask = cv2.morphologyEx(hsvmask, cv2.MORPH_OPEN, morph_open_kernel)
        hsvmask_asmask = cv2.dilate(hsvmask_asmask,dilate_kernel_for_cb_mask,iterations = 1)        
        hsvmask_asmask = cv2.dilate(hsvmask_asmask,dilate_kernel_for_cb_mask,iterations = 1)     

        color_or_bw = cv2.bitwise_or(frame_bw, hsvmask, mask=hsvmask_asmask)   
    else:
        color_or_bw = cv2.bitwise_or(frame_bw, hsvmask)  

    frame_bw = cv2.morphologyEx(frame_bw, cv2.MORPH_OPEN, morph_open_kernel)
    color_or_bw = cv2.dilate(color_or_bw,dilate_kernel_for_cb_mask,iterations = 1)
    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_11color_or_bw.jpg'), color_or_bw)

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

    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_12fillPoly.jpg'), color_or_bw)



    # find contour again after filling polygon
    image, contours, hierarchy =cv2.findContours(color_or_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    frame_bw_color = cv2.cvtColor(color_or_bw, cv2.COLOR_GRAY2BGR)

    obj_id=0
    for cnt in contours : 
        color = colors[int(obj_id) % len(colors)]
        cv2.drawContours(frame_bw_color, cnt, -1, color, 5)
        obj_id=obj_id+1

    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_13findContours.jpg'), frame_bw_color)
    
    # Searching through every region selected to  
    # find the required polygon. 

    obj_id=0
    good_contours=[]
    approximg= img2.copy()
    minrectimg= img2.copy()
    hull = []


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
        color2 = colors[(int(obj_id)+1) % len(colors)]
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        _,_,ww,hh = cv2.boundingRect(cnt)
        # Shortlisting the regions based on there area. 
        #Isconvex = cv2.isContourConvex(cnt)

        # use area to do basic filtering
        # in survelliance camera, due to viewing angle, ww>hh
        if area > area_lower_threshold and area < area_upper_threshold and ((ww>hh*2 and i is 0) or i is 1):  

            # hull.append(cv2.convexHull(cnt, False))   
            # cv2.drawContours(approximg, hull, 0, color, 5)      
            approxPolyDP_epsilon = 0.05*cv2.arcLength(cnt,True)
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


            if len(approx) > length_approx_threshold[0] and len(approx) < length_approx_threshold[1]:
                approx = simplify_contour(cnt, 4)
                perimeter = cv2.arcLength(approx,True)
                ratio = math.sqrt(area)/(perimeter+1e-10)                
                
                if ratio > area_to_perimeter_ratio_lower_threshold and ratio < area_to_perimeter_ratio_upper_threshold:

                    cv2.drawContours(approximg, [cnt], 0, color2, 1) 
                    cv2.drawContours(minrectimg, [cnt], 0, color2, 1) 

                    # if ortho, use contour after minAreaRect seems better
                    # if i is 1 and len(approx) is 4:
                    #     print("approx b4: ", approx)
                    #     rect = cv2.minAreaRect(approx) 
                    #     box = cv2.boxPoints(rect) 
                    #     box = np.int0(box)

                    #     for lyn in range(0, 4):
                    #         approx[lyn][0][0] = box[lyn][0]
                    #         approx[lyn][0][1] = box[lyn][1]
                        
                    #     print("approx after: ", approx)

                    print("obj_id: ", obj_id, " center: (", cX, ", ", cY, ") area: ", area, " perimeter: ", perimeter, "len(approx):", len(approx), " ratio: ", ratio)            
                    cv2.drawContours(approximg, [approx], 0, color, 1) 
                    good_contours.append(approx)

                    x,y,w,h = cv2.boundingRect(approx)
                    m = boundingrectmargin[i]
                    centerx = cX-x+m
                    centery = cY-y+m

                    imgg = frame_gray[max(0, y-m): min(y+h+2*m, frame_gray.shape[0]), max(0, x-m):min(x+w+2*m, frame_gray.shape[1])]
                    #cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_gray.jpg'), imgg)
                    imgg_color = img2[max(0, y-m): min(y+h+2*m, frame_gray.shape[0]), max(0, x-m):min(x+w+2*m, frame_gray.shape[1])]           
                    #cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_color.jpg'), imgg_color)

                    imgg_fs = imgg_color.copy()
                    squares_contours = find_squares(imgg_fs)
                    #print("squares_contours: ", squares_contours)
                    cv2.drawContours(imgg_fs, squares_contours, -1, (0,255,0), 3)
                    # for iy in range(corners.shape[0]):
                    #     cv2.circle(imgg_fs, (corners[i,0,0], corners[i,0,1]), 3, (0,255,0), cv2.FILLED)                    
                    #cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_fs.jpg'), imgg_fs)                      

                    # Parameters for Shi-Tomasi algorithm
                    qualityLevel = 0.01
                    minDistance = 10
                    blockSize = 5
                    gradientSize = 3
                    useHarrisDetector = False
                    k = 0.04
                    maxCorners = 50


                    corners = cv2.goodFeaturesToTrack(imgg, maxCorners, qualityLevel, minDistance, None, \
                        blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)
                    imgg_gftt = imgg_color.copy()
                    for iy in range(corners.shape[0]):
                        cv2.circle(imgg_gftt, (corners[i,0,0], corners[i,0,1]), 3, (0,255,0), cv2.FILLED)                    
                    #cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_gftt.jpg'), imgg_gftt)  

                    edges = cv2.Canny(imgg,100,200, 7)
                    #cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_canny.jpg'), edges)  

                    # Harris corner detector
                    img_harris = imgg_color.copy()
                    blockSize = 2
                    apertureSize = 3
                    k = 0.04
                    thresh = 100
                    # Detecting corners
                    dst = cv2.cornerHarris(imgg, blockSize, apertureSize, k)
                    # Normalizing
                    dst_norm = np.empty(dst.shape, dtype=np.float32)
                    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
                    # Drawing a circle around corners
                    for iii in range(dst_norm.shape[0]):
                        for jjj in range(dst_norm.shape[1]):
                            if int(dst_norm[iii,jjj]) > thresh:
                                cv2.circle(img_harris, (jjj,iii), 5, (0,255,0), 2)

                    #cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + 'harris.jpg'), img_harris)   
                    

                    img_lsd = imgg_color.copy()
                    img_lsd_filter = imgg_color.copy()
                    #Create default parametrization LSD
                    lsd = cv2.createLineSegmentDetector(0)
                    #lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV, 1.0, )

                    #Detect lines in the image
                    lineslsd = lsd.detect(imgg)[0] #Position 0 of the returned tuple are the detected lines
                    img_lsd = lsd.drawSegments(img_lsd,lineslsd)
                    print("len(lineslsd)", len(lineslsd))
                    #print("lineslsd: ", lineslsd)

                    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_lsd.jpg'), img_lsd)   
                    lineslsd_filtered=[]
                    for line in lineslsd:
                        length = linelength(line[0])
                        if length > 15:
                            lineslsd_filtered.append(line)

                    #print("len(lineslsd_filtered)", len(lineslsd_filtered))
                    #Draw detected lines in the image

                    if len(lineslsd_filtered)>0:
                        img_lsd_filter = lsd.drawSegments(img_lsd_filter,np.asarray(lineslsd_filtered))
                        #cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_lsd_filter.jpg'), img_lsd_filter)   


                    img_lsd_merge = imgg_color.copy()
                    line_merged = lines_merging(lineslsd)

                    #print("len(line_merged)", len(line_merged))
                    #print("line_merged: ", np.asarray(line_merged))
                    if len(line_merged)>0:
                        #img_lsd_merge = lsd.drawSegments(img_lsd_merge,np.asarray(line_merged))
                        for merge_line in line_merged:
                            cv2.line(img_lsd_merge,(int(merge_line[0][0]), int(merge_line[0][1])),(int(merge_line[0][2]), int(merge_line[0][3])),(0,0,255),1)  

                        cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_lsd_merge.jpg'), img_lsd_merge)   

                    line_merged_nms=[]
                    for merge_line in line_merged:
                        isRepeat=False
                        for merge_line2 in line_merged:
                            if not merge_line is merge_line2 and checksameline(merge_line[0], merge_line2[0]):
                                isRepeat = True

                        if not isRepeat:
                            line_merged_nms.append(merge_line)
                        else:
                            isExist=False
                            for linee in line_merged_nms:
                                if checksameline(linee[0], merge_line[0]):
                                    isExist=True
                            
                            if not isExist:
                                line_merged_nms.append(merge_line)

                    print("len(line_merged_nms)", len(line_merged_nms))
                    #print("line_merged_nms: ", np.asarray(line_merged_nms))

                    img_lsd_merge_thd_list=[]
                    img_lsd_merge_thd = imgg_color.copy()
                    for merge_line in line_merged_nms:
                        length = linelength(merge_line[0])
                        if length > mergedlineminthreshold[i]:  
                            img_lsd_merge_thd_list.append(merge_line)
                            cv2.line(img_lsd_merge_thd,(int(merge_line[0][0]), int(merge_line[0][1])),(int(merge_line[0][2]), int(merge_line[0][3])),(0,0,255),1)  

                        cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_lsd_merge_thd.jpg'), img_lsd_merge_thd)                                                        
                    print("len(img_lsd_merge_thd_list)", len(img_lsd_merge_thd_list))

                    img_lsd_merge_intersect = imgg_color.copy()
                    convexhullpointlist=[]
                    idx1=0
                    if len(line_merged_nms)>0:
                        #img_lsd_merge = lsd.drawSegments(img_lsd_merge,np.asarray(line_merged))
                        for merge_line in line_merged_nms:
                            idx2=0
                            length = linelength(merge_line[0])
                            if length > mergedlineminthreshold[i]:  
                                for merge_line2 in line_merged_nms:
                                    length2 = linelength(merge_line2[0])        
                                    if length2 > mergedlineminthreshold[i]:  
                                        if idx2 > idx1 and not checksameline(merge_line[0], merge_line2[0]):  
                                            #print("Line for intersect: ", merge_line[0], merge_line2[0])
                                            (xi, yi, valid, r, s) = intersect(merge_line[0], merge_line2[0])   
                                            if valid == 1 and min(math.fabs(r-1.0), math.fabs(r)) < 0.8 and min(math.fabs(s-1.0), math.fabs(s)) < 0.8:
                                                convexhullpointlist.append([[xi, yi]])
                                                cv2.circle(img_lsd_merge_intersect, (int(xi), int(yi)), 5, (0,255,0), 2)

                                    idx2=idx2+1

                            idx1=idx1+1

                        cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_lsd_mintersect.jpg'), img_lsd_merge_intersect)                       


                    img_lsd_convexhull = imgg_color.copy()
                    #print("np.asarray(convexhullpointlist)", np.asarray(convexhullpointlist))
                    if len(convexhullpointlist)>0:
                        hull = cv2.convexHull(np.asarray(convexhullpointlist, dtype=np.float32))
                        print("len(hull)", len(hull))
                        for pt in hull:
                            #print("pt: ", pt)
                            cv2.circle(img_lsd_convexhull, (int(pt[0][0]), int(pt[0][1])), 5, (0,255,0), 2)
                        cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_lsd_mzconvexhull.jpg'), img_lsd_convexhull)  
                    
                    
                    if IsUsingRefinedCorner and len(hull) > 0:
                        hull_in_globalcoord=[]
                        for pt in hull:
                            hull_in_globalcoord.append([[int(pt[0][0]+x-m), int(pt[0][1]+y-m)]])

                        approx_new=[]
                        

                        for pt in approx:
                            min_dist=9999
                            min_dist_index=0
                            idxx=0

                            for pt2 in hull_in_globalcoord:
                                dist = np.linalg.norm(pt[0]-pt2[0])
                                #dist = math.sqrt(math.pow((pt[0][0]-pt2[0][0]-x+m),2)+math.pow((pt[0][1]-pt2[0][1]-y+m),2))
                                #print("dist: ", dist)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_dist_index = idxx
                                idxx=idxx+1

                            #print("min_dist: ", min_dist)
                            if min_dist < RefinedCornerThreshold[i]:
                                print("original pt ", pt, "is refined by ", hull_in_globalcoord[min_dist_index], " with dist=", min_dist)
                                approx_new.append(hull_in_globalcoord[min_dist_index])
                            else:
                                print("No replacement, min dist=", min_dist)
                                approx_new.append(pt)
                        
                        print("old approx: ", approx.flatten())
                        print("new approx: ", np.asarray(approx_new).flatten())
                        approx = np.asarray(approx_new)


                    houghpic = imgg_color.copy()
                    houghparallelpic = imgg_color.copy()
                    lines=None
                    lines_in_2p=[]
                    lines = cv2.HoughLines(edges,2,(5.0*np.pi)/180,60)

                    if not lines is None:
                        for aaa in range(0,lines.shape[0]):
                            rho,theta = lines[aaa,0]

                            a = np.cos(theta)
                            b = np.sin(theta)
                            x0 = a*rho
                            y0 = b*rho
                            x1 = int(x0 + 1000*(-b))
                            y1 = int(y0 + 1000*(a))
                            x2 = int(x0 - 1000*(-b))
                            y2 = int(y0 - 1000*(a))

                            cv2.line(houghpic,(x1,y1),(x2,y2),(0,0,255),2)  
                            lines_in_2p.append([x1, y1, x2, y2])  

                        #cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_hough.jpg'), houghpic)      

                    if not lines is None:
                        for aaa in range(0,lines.shape[0]):
                            rho,theta = lines[aaa,0]

                            for bbb in range(0,lines.shape[0]):
                                rho2,theta2 = lines[bbb,0]

                                #if not aaa is bbb and math.fabs(theta-theta2) < 0.1 and math.fabs(rho-rho2) > 5:
                                if not aaa is bbb and math.fabs(math.fabs(theta-theta2)-1.5708) < 0.05:

                                    (xi, yi, valid, r, s) = intersect(lines_in_2p[aaa], lines_in_2p[bbb])
                                    if valid == 1 and math.fabs(xi-centerx)< 3 and math.fabs(yi-centery)< 3 :

                                        cv2.line(houghparallelpic,(lines_in_2p[aaa][0],lines_in_2p[aaa][1]),(lines_in_2p[aaa][2],lines_in_2p[aaa][3]),(0,0,255),2)    
                                        break

                        #cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_hough_onlycross.jpg'), houghparallelpic)                              


                    im, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    imgcon = imgg_color.copy()
                    cnttt_id=0
                    for cnttt in contours:
                        colorrr = colors[int(cnttt_id) % len(colors)]
                        cv2.drawContours(imgcon, contours, -1, colorrr, 1)
                        cnttt_id=cnttt_id+1
                    #cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_ctr_' + str(cX) + '_' + str(cY) + '_contour.jpg'), imgcon)   



                    result_contour_img[i].append(imgg)
                    result_contour_id[i].append(obj_id)
                    result_contour_cnt[i].append(approx)
                    result_contour_ROI[i].append((x, y, w, h))
                
                else:
                    print("one contour rejected by area_to_perimeter_ratio, which ratio=", ratio, "_obj_id: ", obj_id, " center: (", cX, ", ", cY, ") area: ", area, " =================")
                
            else:
                print("one contour rejected by len(approx), which len(approx)=", len(approx), "_obj_id: ", obj_id, " center: (", cX, ", ", cY, ") area: ", area, "==================")

        obj_id=obj_id+1

    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_14approxPolyDP.jpg'), approximg)
    print("good contour number: ", len(result_contour_id[i]))

    outputmask = np.zeros(color_or_bw.shape, dtype="uint8") 
    for cnt in good_contours : 
        cv2.drawContours(outputmask, [cnt], 0, (255,255,255), -1) 

    outputmask = cv2.dilate(outputmask,dilate_kernel_for_cb_mask,iterations = 2)
    outputmask = cv2.resize(outputmask, (0,0), fx=scaledownratio[i], fy=scaledownratio[i])
    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_15outputmask.jpg'), outputmask)

    # raw = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    # outputmask = cv2.cvtColor(outputmask, cv2.COLOR_GRAY2BGR ) 
    # raw_masked = raw*outputmask
    # cv2.imwrite(os.path.join(output_dir, 'round' + str(i) + '_raw_mask.jpg'), raw_masked)


    obj_id=0
    for cnt in good_contours : 
        color = colors[int(obj_id) % len(colors)]
    
        #print("obj_id: ", obj_id, ", approxcnt: ", cnt)
        rect = cv2.minAreaRect(cnt) 
        box = cv2.boxPoints(rect) 
        box = np.int0(box)
        cv2.drawContours(minrectimg, [box], 0, color, 1)

        obj_id=obj_id+1

    # Showing the image along with outlined arrow. 
    #cv2.imshow('image2', img2)  
    cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_16detected.jpg'), minrectimg)


print("=====================================================================")
print("Matching stage")

# print("Method1: cv2.matchShapes")

# dist_matrix = np.zeros((len(result_contour_img[0]),len(result_contour_img[1])),dtype=np.float32)
# for d,img1 in enumerate(result_contour_img[0]):
#     for t,img2 in enumerate(result_contour_img[1]):
#         # with itself, score=0
#         dist_matrix[d,t] = cv2.matchShapes(img1,img2,cv2.CONTOURS_MATCH_I1,0)

# print("dist_matrix: ", dist_matrix)
# matched_indices = linear_assignment(dist_matrix)
# print("matched_indices: ", matched_indices)

# for matches in matched_indices:
#     print(result_contour_id[0][matches[0]], result_contour_id[1][matches[1]])


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
ransacReprojThreshold=15

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
    bestcopylist=None
    copylist= result_contour_cnt[1][matches[1]]


    print("Guessing delta_idx for matches: ", result_contour_id[0][matches[0]], result_contour_id[1][matches[1]])
    

    # calculate the reprojection error of the center of mass, which has a unique answer
    for roundj in range(0, 4):

        if roundj is not 0:
            endvalue = copylist[-1]
            copylist = np.insert(copylist[:-1], 0, endvalue, axis=0)

        print("copylist: ", copylist.flatten())

        #homography, mask = cv2.findHomography(result_contour_cnt[0][matches[0]], copylist, cv2.RANSAC,ransacReprojThreshold)
        mixed_src = np.concatenate((result_contour_cnt[0][matches[0]], src_pts))
        mixed_dst = np.concatenate((copylist, dst_pts))
        #homography, mask = cv2.findHomography(result_contour_cnt[0][matches[0]], copylist, 0)
        homography, mask = cv2.findHomography(mixed_src, mixed_dst, 0)
        #print("mask: ", mask.transpose())

   
        reprojectimgpts = cv2.perspectiveTransform(src_pts, homography)
        reprojectionerror=0
        errorlist=[]

        for zz in range(0, len(src_pts)):
            error = math.sqrt(math.pow(reprojectimgpts[zz][0][0]-dst_pts[zz][0][0], 2)+math.pow(reprojectimgpts[zz][0][1]-dst_pts[zz][0][1], 2))
            errorlist.append(error)
            reprojectionerror = reprojectionerror + error

        print("errorlist=", errorlist)
        print("reprojectionerror: ", reprojectionerror)
        if reprojectionerror < besterror:
            besterror = reprojectionerror
            bestidx = roundj
            bestcopylist = copylist.copy()


    delta_idx[counttt] = bestidx
    print("bestidx: ", bestidx)
    print("besterror: ", besterror)
    print("===============================================================================")
    counttt=counttt+1

    if final_src_pts is None:
        final_src_pts = np.asarray(result_contour_cnt[0][matches[0]])
    else:
        final_src_pts = np.append(final_src_pts, result_contour_cnt[0][matches[0]], axis=0)


    if final_dst_pts is None:
        final_dst_pts = np.asarray(bestcopylist.copy())
    else:
        final_dst_pts = np.append(final_dst_pts, bestcopylist, axis=0)


print("delta_idx: ", delta_idx)
#print("final_dst_pts: ", final_dst_pts)

#print("final_src_pts: ", final_src_pts*scaledownratio[0])
#print("final_dst_pts: ", final_dst_pts*scaledownratio[1])


print("Input from src_dst_test_points.yml...")
print("[INPUT]")
fs_read = cv2.FileStorage("src_dst_test_points.yml", cv2.FILE_STORAGE_READ)
cross_subpixel_matrix = fs_read.getNode("cross_subpixel_matrix").mat()
print("cross_subpixel_matrix: ", cross_subpixel_matrix)



# Print out resized rgb image with the point index to see if the matching is correct
displayimg2 = frame_img2[0].copy()
for jjj in range(0, len(final_src_pts)):
    cv2.putText(displayimg2,str(jjj), (int(final_src_pts[jjj][0][0]) ,int(final_src_pts[jjj][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.circle(displayimg2,(int(final_src_pts[jjj][0][0]) ,int(final_src_pts[jjj][0][1])), 5, (0, 255, 0))
#cv2.imwrite(os.path.join(output_dir, 'i=' + str(0) + '_raw_color_matched_index.jpg'), displayimg2)

# Also plot the test points to double confirm
for qqyy in range(0, cross_subpixel_matrix.shape[0]):
    cv2.circle(displayimg2,(int(cross_subpixel_matrix[qqyy][1]/scaledownratio[0]) ,int(cross_subpixel_matrix[qqyy][2]/scaledownratio[0])), 5, (255, 255, 255))

# also plot the test src_pts
for qqyy in range(0, len(src_pts)):
    cv2.circle(displayimg2,(int(src_pts[qqyy][0][0]) ,int(src_pts[qqyy][0][1])), 5, (0, 0, 255), -1)    


displayimg3 = frame_img2[1].copy()
for jjj in range(0, len(final_dst_pts)):
    cv2.putText(displayimg3,str(jjj), (int(final_dst_pts[jjj][0][0]), int(final_dst_pts[jjj][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.circle(displayimg3,(int(final_dst_pts[jjj][0][0]) ,int(final_dst_pts[jjj][0][1])), 5, (0, 255, 0))
#cv2.imwrite(os.path.join(output_dir, 'i=' + str(1) + '_raw_color_matched_index.jpg'), displayimg2)


# also plot the test src_pts
for qqyy in range(0, len(dst_pts)):
    cv2.circle(displayimg3,(int(dst_pts[qqyy][0][0]) ,int(dst_pts[qqyy][0][1])), 5, (0, 0, 255), -1)    

# calculate the px to px homography for visualization
pxhomography, pxmask = cv2.findHomography(final_src_pts*scaledownratio[0], final_dst_pts*scaledownratio[1], 0)
print("pxmask: ", pxmask.transpose())
print("pxhomography: ", pxhomography)

dst_px = cv2.perspectiveTransform((cross_subpixel_matrix[:, 1:3]).reshape(-1,1,2).astype(np.float64), pxhomography)

# Also plot the test points to double confirm
for qqxx in range(0, cross_subpixel_matrix.shape[0]):
    cv2.circle(displayimg3,(int(dst_px[qqxx][0][0]/scaledownratio[1]) ,int(dst_px[qqxx][0][1]/scaledownratio[1])), 5, (255, 255, 255))


h1, w1 = displayimg2.shape[:2]
h2, w2 = displayimg3.shape[:2]

#create empty matrix
vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

#combine 2 images
vis[:h1, :w1,:3] = displayimg2
vis[:h2, w1:w1+w2,:3] = displayimg3

cv2.imwrite(os.path.join(output_dir, 'zzz_raw_color_matched_index.jpg'), vis)

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

homography, mask = cv2.findHomography(final_src_pts*scaledownratio[0], final_dst_pts_gps, cv2.RANSAC,1e-4)

print("mask: ", mask.transpose())
print("homography: ", homography)
print("Storing the result in homography.yml...")
fs_write = cv2.FileStorage('homography.yml', cv2.FILE_STORAGE_WRITE)
fs_write.write("homography_matrix", homography)
fs_write.release()

print('Whole process took seconds: ',time.time() - start)
   
# Exiting the window if 'q' is pressed on the keyboard. 
if cv2.waitKey(0) & 0xFF == ord('q'):  
    cv2.destroyAllWindows() 

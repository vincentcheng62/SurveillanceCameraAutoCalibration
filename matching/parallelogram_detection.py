import numpy as np 
import cv2 
import time, datetime
import os
import math

print("cv2.__version__: ", cv2.__version__)
   
###### Input parameters ######
print("########## Input parameters ##########")

IsGeoTiff=True
MarkerSize=1.5 # in meter
orthomosaicscale=6.66  # mm/px, shown in photoscan UI
marker_area_variation=0.2  # 20%
area_to_perimeter_ratio_threshold=0.1 # 20%, for square it should be (+-0.25)

#img_path='real2.jpg'
#img_path='drone_ch.jpg'
img_path='marker.jpg'
#img_path='marker6m.jpg'

thrhd_grayvalue_high_for_whitepart_of_chessboard=170
thrhd_grayvalue_low_for_whitepart_of_chessboard=50
morph_open_kernel_radius_in_px=5
morph_close_kernel_radius_in_px=10
dilate_kernel_size_in_px_for_cb_mask_radius=6
dilate_kernel_size_for_white=60
approxPolyDP_epsilon=20
scaledownratio = 3.0



print("IsGeoTiff: ", IsGeoTiff)
print("MarkerSize(in m): ", MarkerSize)
print("orthomosaicscale(mm/px): ", orthomosaicscale)
print("img_path: ", img_path)
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
side_in_px = (MarkerSize*1000)/(orthomosaicscale*scaledownratio)
print("expected marker side in px (in resized img): ", side_in_px)
area_in_px = side_in_px*side_in_px
print("expected marker area in px (in resized img): ", area_in_px)


# Reading image 
img2 = cv2.imread(img_path, cv2.IMREAD_COLOR) 
print("input image original size: ", img2.shape)

img2 = cv2.resize(img2, (0,0), fx=(1/scaledownratio), fy=(1/scaledownratio)) 
print("input image resized size: ", img2.shape)

# convert to HSV

imghsv = img2.copy()
imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2HSV ) 



   
# Reading same image in another variable and  
# converting to gray scale. 
frame_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
frame_gray = cv2.resize(frame_gray, (0,0), fx=(1/scaledownratio), fy=(1/scaledownratio)) 

# set 255 to 0 since orthomosaic has white boundary
for y in range(len(frame_gray)):
    for x in range(len(frame_gray[0])):
        if frame_gray[y][x] == 255:
            frame_gray[y][x] = 0


hsvmask=None
for i in range(0, len(lowerhsvlist)):
    newhsvmask = cv2.inRange(imghsv, lowerhsvlist[i], upperhsvlist[i])
    cv2.imwrite(os.path.join(output_dir, 'parallelogram_after' + str(i) + 'hsv.jpg'), newhsvmask)

    if hsvmask is None:
        hsvmask = newhsvmask.copy()
    else:
        hsvmask = cv2.bitwise_or(newhsvmask, hsvmask)   

hsvmask = cv2.morphologyEx(hsvmask, cv2.MORPH_CLOSE, morph_open_kernel)
cv2.imwrite(os.path.join(output_dir, 'parallelogram_1after_color_seg.jpg'), hsvmask)





##### old #####


bw_ret,frame_bwh_original = cv2.threshold(frame_gray,thrhd_grayvalue_high_for_whitepart_of_chessboard,255,cv2.THRESH_BINARY)
frame_bwh = cv2.morphologyEx(frame_bwh_original, cv2.MORPH_OPEN, morph_open_kernel)
frame_bwh = cv2.dilate(frame_bwh,dilate_kernel_for_cb_mask,iterations = 1)



cv2.imwrite(os.path.join(output_dir, 'parallelogram_2threshold_w.jpg'), frame_bwh)


frame_bwh_dilate = cv2.dilate(frame_bwh,dilate_kernel_white,iterations = 1)


cv2.imwrite(os.path.join(output_dir, 'parallelogram_3dilate_w_as_mask.jpg'), frame_bwh_dilate)


# set 255 to 0 since orthomosaic has white boundary
for y in range(len(frame_gray)):
    for x in range(len(frame_gray[0])):
        if frame_gray[y][x] == 0:
            frame_gray[y][x] = 255

bw_ret,frame_bwl = cv2.threshold(frame_gray,thrhd_grayvalue_low_for_whitepart_of_chessboard,255,cv2.THRESH_BINARY_INV)
frame_bwl = cv2.morphologyEx(frame_bwl, cv2.MORPH_CLOSE, morph_close_kernel)
frame_bwl = cv2.dilate(frame_bwl,dilate_kernel_for_cb_mask,iterations = 1)

cv2.imwrite(os.path.join(output_dir, 'parallelogram_4threshold_dark.jpg'), frame_bwl)

frame_bw = cv2.bitwise_or(frame_bwh_original, frame_bwl, mask=frame_bwh_dilate)   

cv2.imwrite(os.path.join(output_dir, 'parallelogram_5white_OR_dark.jpg'), frame_bw)


##### old #####

color_or_bw = cv2.bitwise_or(frame_bw, hsvmask)  

frame_bw = cv2.morphologyEx(frame_bw, cv2.MORPH_OPEN, morph_open_kernel)
color_or_bw = cv2.dilate(color_or_bw,dilate_kernel_for_cb_mask,iterations = 1)
cv2.imwrite(os.path.join(output_dir, 'parallelogram_6color_or_bw.jpg'), color_or_bw)

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
image, contours, hierarchy =cv2.findContours(color_or_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
for cnt in contours : 
    cv2.fillPoly(color_or_bw, pts =[cnt], color=(255,255,255))

cv2.imwrite(os.path.join(output_dir, 'parallelogram_7fillPoly.jpg'), color_or_bw)



# find contour again after filling polygon
image, contours, hierarchy =cv2.findContours(color_or_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

frame_bw_color = cv2.cvtColor(color_or_bw, cv2.COLOR_GRAY2BGR)

obj_id=0
for cnt in contours : 
    color = colors[int(obj_id) % len(colors)]
    cv2.drawContours(frame_bw_color, cnt, -1, color, 5)
    obj_id=obj_id+1

cv2.imwrite(os.path.join(output_dir, 'parallelogram_8findContours.jpg'), frame_bw_color)
   
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

if IsGeoTiff:
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

        print("len(approx): ", len(approx))

        count=1
        while not len(approx) is 3:
            if len(approx)>3:
                approx = cv2.approxPolyDP(cnt,  approxPolyDP_epsilon*(1+count*0.1), False) 
            elif len(approx)<3:
                approx = cv2.approxPolyDP(cnt,  approxPolyDP_epsilon*(1-count*0.1), False) 
                if count >= 10:
                    break

            count=count+1
            print("count: ", count, " len(approx): ", len(approx))


        print("approx: ", approx)
        perimeter = cv2.arcLength(approx,True)
        ratio = math.sqrt(area)/(perimeter+1e-10)

        #if len(approx) > 2 and len(approx) < 6 and ratio > 0.15:
        #if ratio > area_to_perimeter_ratio_lower_threshold and ratio < area_to_perimeter_ratio_upper_threshold:
        print("center: (", cX, ", ", cY, ") area: ", area, " perimeter: ", perimeter, "len(approx):", len(approx), " ratio: ", ratio)            
        cv2.drawContours(approximg, [approx], 0, color, 5) 
        good_contours.append(approx)

    obj_id=obj_id+1

cv2.imwrite(os.path.join(output_dir, 'parallelogram_9approxPolyDP.jpg'), approximg)

outputmask = np.zeros(color_or_bw.shape, dtype="uint8") 
for cnt in good_contours : 
    cv2.drawContours(outputmask, [cnt], 0, (255,255,255), -1) 

outputmask = cv2.dilate(outputmask,dilate_kernel_for_cb_mask,iterations = 2)
outputmask = cv2.resize(outputmask, (0,0), fx=scaledownratio, fy=scaledownratio)
cv2.imwrite(os.path.join(output_dir, 'outputmask.jpg'), outputmask)

raw = cv2.imread(img_path, cv2.IMREAD_COLOR) 
outputmask = cv2.cvtColor(outputmask, cv2.COLOR_GRAY2BGR ) 
raw_masked = raw*outputmask
cv2.imwrite(os.path.join(output_dir, 'raw_mask.jpg'), raw_masked)


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
cv2.imwrite(os.path.join(output_dir, 'parallelogram_10detected.jpg'), img2)

print('Whole process took seconds: ',time.time() - start)
   
# Exiting the window if 'q' is pressed on the keyboard. 
if cv2.waitKey(0) & 0xFF == ord('q'):  
    cv2.destroyAllWindows() 
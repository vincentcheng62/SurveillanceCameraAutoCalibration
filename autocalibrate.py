import numpy as np
import cv2 as cv
import cv2
import time, datetime
from matplotlib import pyplot as plt
import operator
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linspace
import argparse
from imutils.object_detection import non_max_suppression
from numpy.linalg import inv
from math import log10, floor
import pyproj

#cap = cv.VideoCapture('rtsp://admin:h0940232@172.18.9.100/Streaming/Channels/1')
cap = cv.VideoCapture('rtsp://admin:h0940232@192.168.0.100/Streaming/Channels/1')
#cap = cv.VideoCapture('output2018-12-28-11-50-17.avi')
square_size = 0.375
IsDetectPeople = False
IsSaveToVideo = True
IsEachFrameDebug = False
IsPlottingPtGrid = True
hog_threshold = 0.9
regionpts = np.array([[361,195],[414,195],[558,691],[761,685]], np.int32)


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv

def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    M[1,1] = 0
    M[1,2] = 1
    M[2,1] = -1
    M[2,2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))

def round_to_1(x, sig=2):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)

def GetWindowWithAxis(Size_of_w, physical_size):
    int_size = int(Size_of_w*0.5)
    center = (int_size, int_size)
    px_of_meter = int(Size_of_w/physical_size)
    img = np.zeros((Size_of_w, Size_of_w, 3), np.uint8)
    img = cv.line(img, center, (center[0], center[1]+px_of_meter), (0,0,255), 3)
    img = cv.line(img, center, (center[0]-px_of_meter, center[1]), (0,255,0), 3)
    return img

def PrintLocalization(Lmap, bigger_frame, pick, ratio, Size_of_w, physical_size, camera_matrix_manual, dist_coefs_manual, ref_rvec, ref_tvec, calib_corners):
    Lmap_localized = Lmap.copy()
    bigger_frame_reproject = bigger_frame.copy()

    px_of_meter = 1.0*Size_of_w/physical_size
    rot, jaco = cv.Rodrigues(ref_rvec)
    #ext = np.vstack((np.hstack((rot, ref_tvec)), np.array([0, 0, 0, 1])))
    #print(ext)
    #ext_inv = inv(ext)
    human_height=1.7
    for (xA, yA, xB, yB) in pick:
        head_pt = [(xA+xB)*0.5*ratio, min(yA, yB)*ratio] # (u1, v1)
        bottom_pt = [(xA+xB)*0.5*ratio, max(yA, yB)*ratio] # (u2, v2)
        bigger_frame_reproject = cv.circle(bigger_frame_reproject, (int(bottom_pt[0]), int(bottom_pt[1])), 5, (0,255,0), thickness=3, lineType=8, shift=0) 
        bigger_frame_reproject = cv.circle(bigger_frame_reproject, (int(head_pt[0]), int(head_pt[1])), 5, (0,255,0), thickness=3, lineType=8, shift=0) 

        print("head_pt: ", head_pt)
        print("bottom_pt: ", bottom_pt)
        bp = np.zeros((1, 2), np.float32)
        bp[0][0] = bottom_pt[0]
        bp[0][1] = bottom_pt[1]
        hp = np.zeros((1, 2), np.float32)
        hp[0][0] = head_pt[0]
        hp[0][1] = head_pt[1]        
        #print("calib_corners: ", calib_corners[0])
        dst = cv.undistortPoints(bp.reshape(-1,1,2).astype(np.float32), camera_matrix_manual, dist_coefs_manual)
        bottom_pt = dst[0][0]
        dst2 = cv.undistortPoints(hp.reshape(-1,1,2).astype(np.float32), camera_matrix_manual, dist_coefs_manual)        
        head_pt = dst2[0][0]
        # according to the formula: s1=1.7*r33+s2, s2=1.7*(r23-v1*r33)/(v1-v2)
        #s2 = 1.7*(rot[1][2]-head_pt[1]*rot[2][2])/(head_pt[1]-bottom_pt[1])
        print("head_pt after undistort: ", head_pt)
        print("bottom_pt after undistort: ", bottom_pt)
        #bottom_pt_center_normalized = (bottom_pt[0]-camera_matrix_manual[0][2], bottom_pt[1]-camera_matrix_manual[1][2])
        #print("bottom_pt_center_normalized: ", bottom_pt_center_normalized)        
        #s1 = 1.7*rot[2][2]+s2

        #print("ref_tvec: ", ref_tvec[0][0], ref_tvec[1][0], ref_tvec[2][0])
        #AlgMethod=1  # 0 : assume z=0, 1: assume height=1.7
        s1g, s2g = GetBottomPtScale(head_pt, bottom_pt, rot, ref_tvec, 0)
        s1, s2 = GetBottomPtScale(head_pt, bottom_pt, rot, ref_tvec, 1)

        img_pt_h = np.array([[head_pt[0]], [head_pt[1]], [1]])
        img_pt_b = np.array([[bottom_pt[0]], [bottom_pt[1]], [1]])
        #print("zz: ", s2*img_pt-ref_tvec)
        #print("tra: ", cv.transpose(rot))
        result1 = np.matmul(cv.transpose(rot), s1*img_pt_h-ref_tvec)
        result2 = np.matmul(cv.transpose(rot), s2*img_pt_b-ref_tvec)
        resultg = np.matmul(cv.transpose(rot), s2g*img_pt_b-ref_tvec)
        print("result1: ",cv.transpose(result1))
        print("result2: ",cv.transpose(result2))
        print("resultg: ",cv.transpose(resultg))
        print("resultg-result2: ",cv.transpose(resultg-result2))

        center = (int(Size_of_w*0.5-result2[1][0]*px_of_meter), int(result2[0][0]*px_of_meter+Size_of_w*0.5))     
        #print("world XYZ: ", result2[0][0], result2[1][0], result2[2][0])    
        print("window XY of bottom pt: ", center[0], center[1])
        #print("result: ", result)
        pchar = "[" + str(round_to_1(result2[0][0])) + ", " + str(round_to_1(result2[1][0])) + ", " + str(round_to_1(result2[2][0])) + "]"

        Lmap_localized = cv.circle(Lmap_localized, center, 5, (255,0,0), thickness=3, lineType=8, shift=0) 
        cv.putText(Lmap_localized, pchar, center, cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1) 

        #Project the resulting 3d point onto 2d image again for comfirmation.
        imgpts, jac = cv.projectPoints(np.array([[result2[0][0], result2[1][0], result2[2][0]], [0.0, 0.0, 0.0]]), ref_rvec, ref_tvec, camera_matrix_manual, dist_coefs_manual)
        intpt =  (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
        print("reproject result2 to 2d plane: ", intpt)

        if intpt[0]>0 and intpt[0]<bigger_frame_reproject.shape[1] and intpt[1]>0 and intpt[1]<bigger_frame_reproject.shape[0]:
            bigger_frame_reproject = cv.circle(bigger_frame_reproject, intpt, 5, (255,0,0), thickness=3, lineType=8, shift=0) 
            cv.putText(bigger_frame_reproject, pchar, (intpt[0], intpt[1]+30), cv.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 2)  
        

        # Recover the gps coordinate
        gps_coord = RecoverGPSCoord(resultg)
        print("gps_coord: ", gps_coord)

        print("==============================================================================================")       
    

    #Calculate gps coord of all the other chessboard origin for debug
    for org in OtherCBOrigin2D:
        bigger_frame_reproject = cv.circle(bigger_frame_reproject, (org[0], org[1]), 5, (0,0,255), thickness=3, lineType=8, shift=0) 
        porgchar = "[" + str(org[0]) + ", " + str(org[1]) + "]"
        cv.putText(bigger_frame_reproject, porgchar, (int(org[0]+30), int(org[1])+40), cv.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)  
        orgp = np.zeros((1, 2), np.float32)
        orgp[0][0] = org[0]
        orgp[0][1] = org[1]        
        dst = cv.undistortPoints(orgp.reshape(-1,1,2).astype(np.float32), camera_matrix_manual, dist_coefs_manual)
        orgp = dst[0][0]
        s1g, s2g = GetBottomPtScale([0, 0], orgp, rot, ref_tvec, 0)
        img_pt_org = np.array([[orgp[0]], [orgp[1]], [1]])
        resultorgg = np.matmul(cv.transpose(rot), s2g*img_pt_org-ref_tvec)
        org_gps_coord = RecoverGPSCoord(resultorgg)
        print("CB origin: ", org, ", gps coord: ", org_gps_coord)

    return Lmap_localized, bigger_frame_reproject

def draw_axis_and_ptgrid(img, corners, imgpts, ptgrid):

    cv.drawChessboardCorners(img, pattern_size, corners, True)
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 3)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 3)

    # Also draw a point grid in the z-plane for debug purpose
    if IsPlottingPtGrid:
        for pt in ptgrid:
            img = cv.circle(img, tuple(pt.ravel()), 4, (0,255,0), thickness=2, lineType=8, shift=0) 

    return img

def create_camera_model(camera_matrix, width, height, scale_focal, draw_frame_axis=True):
    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    focal = 2 / (fx + fy)
    f_scale = scale_focal * focal

    print("f_scale: ", f_scale)

    # draw image plane
    X_img_plane = np.ones((4,5))
    X_img_plane[0:3,0] = [-width, height, f_scale]
    X_img_plane[0:3,1] = [width, height, f_scale]
    X_img_plane[0:3,2] = [width, -height, f_scale]
    X_img_plane[0:3,3] = [-width, -height, f_scale]
    X_img_plane[0:3,4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4,3))
    X_triangle[0:3,0] = [-width, -height, f_scale]
    X_triangle[0:3,1] = [0, -2*height, f_scale]
    X_triangle[0:3,2] = [width, -height, f_scale]

    # draw camera
    X_center1 = np.ones((4,2))
    X_center1[0:3,0] = [0, 0, 0]
    X_center1[0:3,1] = [-width, height, f_scale]

    X_center2 = np.ones((4,2))
    X_center2[0:3,0] = [0, 0, 0]
    X_center2[0:3,1] = [width, height, f_scale]

    X_center3 = np.ones((4,2))
    X_center3[0:3,0] = [0, 0, 0]
    X_center3[0:3,1] = [width, -height, f_scale]

    X_center4 = np.ones((4,2))
    X_center4[0:3,0] = [0, 0, 0]
    X_center4[0:3,1] = [-width, -height, f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4,2))
    X_frame1[0:3,0] = [0, 0, 0]
    X_frame1[0:3,1] = [f_scale*2, 0, 0]

    X_frame2 = np.ones((4,2))
    X_frame2[0:3,0] = [0, 0, 0]
    X_frame2[0:3,1] = [0, f_scale*2, 0]

    X_frame3 = np.ones((4,2))
    X_frame3[0:3,0] = [0, 0, 0]
    X_frame3[0:3,1] = [0, 0, f_scale*2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]

def create_board_model(extrinsics, board_width, board_height, square_size, draw_frame_axis=True):
    width = board_width*square_size
    height = board_height*square_size

    # draw calibration board
    X_board = np.ones((4,5))
    #X_board_cam = np.ones((extrinsics.shape[0],4,5))
    X_board[0:3,0] = [0,0,0]
    X_board[0:3,1] = [width,0,0]
    X_board[0:3,2] = [width,height,0]
    X_board[0:3,3] = [0,height,0]
    X_board[0:3,4] = [0,0,0]

    # draw board frame axis
    X_frame1 = np.ones((4,2))
    X_frame1[0:3,0] = [0, 0, 0]
    X_frame1[0:3,1] = [height, 0, 0]

    X_frame2 = np.ones((4,2))
    X_frame2[0:3,0] = [0, 0, 0]
    X_frame2[0:3,1] = [0, height, 0]

    X_frame3 = np.ones((4,2))
    X_frame3[0:3,0] = [0, 0, 0]
    X_frame3[0:3,1] = [0, 0, height]

    if draw_frame_axis:
        return [X_board, X_frame1, X_frame2, X_frame3]
    else:
        return [X_board]

def draw_camera_boards(ax, camera_matrix, cam_width, cam_height, scale_focal,
                       extrinsics, board_width, board_height, square_size,
                       patternCentric):
    min_values = np.zeros((3,1))
    min_values = np.inf
    max_values = np.zeros((3,1))
    max_values = -np.inf

    if patternCentric:
        X_moving = create_camera_model(camera_matrix, cam_width, cam_height, scale_focal)
        #print("X_moving(camera): ", X_moving)
        X_static = create_board_model(extrinsics, board_width, board_height, square_size)
        #print("X_static(board): ", X_static)
    else:
        X_static = create_camera_model(camera_matrix, cam_width, cam_height, scale_focal, True)
        #print("X_static(board): ", X_static)
        X_moving = create_board_model(extrinsics, board_width, board_height, square_size)
        #print("X_moving(camera): ", X_moving)

    cm_subsection = linspace(0.0, 1.0, extrinsics.shape[0])
    colors = [ cm.jet(x) for x in cm_subsection ]

    #Plot the camera
    for i in range(len(X_static)):
        X = np.zeros(X_static[i].shape)
        for j in range(X_static[i].shape[1]):
            X[:,j] = transform_to_matplotlib_frame(np.eye(4), X_static[i][:,j])
        ax.plot3D(X[0,:], X[1,:], X[2,:], color='r')
        #print("printing red pt at:", X[0,:], X[1,:], X[2,:])
        min_values = np.minimum(min_values, X[0:3,:].min(1))
        max_values = np.maximum(max_values, X[0:3,:].max(1))

    #Plot the board
    for idx in range(extrinsics.shape[0]):
        R, _ = cv.Rodrigues(extrinsics[idx,0:3])
        cMo = np.eye(4,4)
        cMo[0:3,0:3] = R
        cMo[0:3,3] = extrinsics[idx,3:6]
        for i in range(len(X_moving)):
            X = np.zeros(X_moving[i].shape)
            for j in range(X_moving[i].shape[1]):
                X[0:4,j] = transform_to_matplotlib_frame(cMo, X_moving[i][0:4,j], patternCentric)
                # if i == 0 and j == 0:
                #     print(X[0:4,j])
            ax.plot3D(X[0,:], X[1,:], X[2,:], color=colors[idx])
            if i==0:
                print(X[:,0])
            min_values = np.minimum(min_values, X[0:3,:].min(1))
            max_values = np.maximum(max_values, X[0:3,:].max(1))

    return min_values, max_values


#cap = cv.VideoCapture('sample.MOV')
#cap = cv2.VideoCapture('rtsp://172.18.9.99/axis-media/media.amp')
#time.sleep(5)
#print(cv2.__version__)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(30,30))
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

IsVanishingCalibration=False

if IsVanishingCalibration:
    cv.namedWindow("frame")
    cv.moveWindow("frame", 40,10)
    cv.namedWindow("fgmask")
    cv.moveWindow("fgmask", 720,10)
    cv.namedWindow("axis")
    cv.moveWindow("axis", 40,420)

line_db_need_to_collect=100000 # set lower for debug purpose
line_db = []
contour_area_min=600

camera_matrix_manual = np.zeros((3, 3), np.float32)
camera_matrix_manual[0, 0] = 1009.60665
#camera_matrix_manual[1, 0] = 0
#camera_matrix_manual[2, 0] = 0
#camera_matrix_manual[0, 1] = 0
camera_matrix_manual[1, 1] = 1009.32417
#camera_matrix_manual[2, 1] = 0
camera_matrix_manual[0, 2] = 651.53609
camera_matrix_manual[1, 2] = 336.868
camera_matrix_manual[2, 2] = 1

dist_coefs_manual = np.zeros((1, 8), np.float32)     
dist_coefs_manual[0, 0] = -6.08059316
dist_coefs_manual[0, 1] = 9.70169024
dist_coefs_manual[0, 2] = 1.60141342e-03
dist_coefs_manual[0, 3] = -6.39510521e-05 
dist_coefs_manual[0, 4] = -1.77135020  
dist_coefs_manual[0, 5] = -5.71916015  
dist_coefs_manual[0, 6] = 7.55768940  
dist_coefs_manual[0, 7] = 1.36953813  

pattern_size = (4, 3)
#square_size = 0.06395



pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
FinishCalibration = False
ref_rvec=None
ref_tvec=None
ptgrid = None
grid = np.float32([[0.5,-0.5,0], [0.5,0,0], [0.5,0.5,0],
                   [0,-0.5,0], [0,0,0], [0,0.5,0],
                   [-0.5,-0.5,0], [-0.5,0,0], [-0.5,0.5,0],
                   [-1,-0.5,0], [-1,0,0], [-1,0.5,0],
                   [-1.5,-0.5,0], [-1.5,0,0], [-1.5,0.5,0],
                   [-2,-0.5,0], [-2,0,0], [-2,0.5,0],
                   [-2.5,-0.5,0], [-2.5,0,0], [-2.5,0.5,0],
                   [-3,-0.5,0], [-3,0,0], [-3,0.5,0],
                   [-3.5,-0.5,0], [-3.5,0,0], [-3.5,0.5,0],
                   [-4,-0.5,0], [-4,0,0], [-4,0.5,0],
                   [-4.5,-0.5,0], [-4.5,0,0], [-4.5,0.5,0],
                   [-5,-0.5,0], [-5,0,0], [-5,0.5,0],
                   [-5.5,-0.5,0], [-5.5,0,0], [-5.5,0.5,0],
                   [-6,-0.5,0], [-6,0,0], [-6,0.5,0],
                   [-6.5,-0.5,0], [-6.5,0,0], [-6.5,0.5,0],
                   [-7,-0.5,0], [-7,0,0], [-7,0.5,0],
                   [-7.5,-0.5,0], [-7.5,0,0], [-7.5,0.5,0],
                   [-8,-0.5,0], [-8,0,0], [-8,0.5,0],
                   [-8.5,-0.5,0], [-8.5,0,0], [-8.5,0.5,0],
                   [-9,-0.5,0], [-9,0,0], [-9,0.5,0],
                   [-9.5,-0.5,0], [-9.5,0,0], [-9.5,0.5,0],
                   [-10,-0.5,0], [-10,0,0], [-10,0.5,0],
                   [-10.5,-0.5,0], [-10.5,0,0], [-10.5,0.5,0],
                   [-11,-0.5,0], [-11,0,0], [-11,0.5,0],
                   [-11.5,-0.5,0], [-11.5,0,0], [-11.5,0.5,0],
                   [-12,-0.5,0], [-12,0,0], [-12,0.5,0],
                   [-12.5,-0.5,0], [-12.5,0,0], [-12.5,0.5,0],
                   [-13,-0.5,0], [-13,0,0], [-13,0.5,0],
                   [-13.5,-0.5,0], [-13.5,0,0], [-13.5,0.5,0],
                   [-14,-0.5,0], [-14,0,0], [-14,0.5,0],
                   [-14.5,-0.5,0], [-14.5,0,0], [-14.5,0.5,0]] ).reshape(-1,3)

cb_to_ecef_transform = None
OtherCBOrigin2D=[]

# def draw(img, corners, imgpts):
#     corner = tuple(corners[0].ravel())
#     img = cv.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 3)
#     img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
#     img = cv.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 3)
#     return img

def GetWindowWithAxis(Size_of_w, physical_size):
    int_size = int(Size_of_w*0.5)
    center = (int_size, int_size)
    px_of_meter = int(Size_of_w/physical_size)
    img = np.zeros((Size_of_w, Size_of_w, 3), np.uint8)
    img = cv.line(img, center, (center[0], center[1]+px_of_meter), (0,0,255), 3)
    img = cv.line(img, center, (center[0]-px_of_meter, center[1]), (0,255,0), 3)
    return img

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


def GetWindowWithAxis(Size_of_w, physical_size):
    int_size = int(Size_of_w*0.5)
    center = (int_size, int_size)
    px_of_meter = int(Size_of_w/physical_size)
    img = np.zeros((Size_of_w, Size_of_w, 3), np.uint8)
    img = cv.line(img, center, (center[0], center[1]+px_of_meter), (0,0,255), 3)
    img = cv.line(img, center, (center[0]-px_of_meter, center[1]), (0,255,0), 3)
    return img

def GetBottomPtScale(head_pt, bottom_pt, rot, ref_tvec, method=0):
    s1=1
    s2=1
    r13=rot[0][2]
    r23=rot[1][2]
    r33=rot[2][2]
    t1=ref_tvec[0][0]
    t2=ref_tvec[1][0]
    t3=ref_tvec[2][0]
    u1=head_pt[0]
    v1=head_pt[1]
    u2=bottom_pt[0]
    v2=bottom_pt[1]
    h=-1.7

    if method==0:
        # Assume the world coord of the bottom pt (X, Y, Z=0), then there is unique solution for s2
        Tz = r13*t1+r23*t2+r33*t3
        dz = r13*u2+r23*v2+r33*1.0
        s2 = Tz/dz
        #print("s2g: ", s2)

    else:
        # Use head-pt and bottom-pt to approximate the (X, Y, Z) of the bottom pt in which Z != 0
        # Need to minimize f where f(s1, s2) = Norm((s1u1-s2u2-hr13, s1v1-s2v2-hr23, s1-s2-hr33))
        # That means, want to find a pair of (s1, s2) such that their world coord (X1, Y1, Z1), (X2, Y2, Z2)
        # can achieve X1 close to X2, Y1 close to Y2, Z1-Z2 close to h=1.7m normal human height
        # Taking df/ds1 and df/ds2=0 and solve for s1 and s2
        # Since u1=u2,...


        Df = np.zeros((2, 2), np.float32)
        Df[0, 0] = 2*(u1*u1+v1*v1+1)
        Df[0, 1] = -2*(u1*u1+v1*v2+1)
        Df[1, 0] = -2*(u1*u1+v1*v2+1)
        Df[1, 1] = 2*(u1*u1+v2*v2+1)
        #print("Df: ", Df)

        Zero = np.zeros((2, 1), np.float32)
        Zero[0, 0] = 2*h*(u1*r13+v1*r23+r33)
        Zero[1, 0] = -2*h*(u1*r13+v2*r23+r33)
        #print("Zero: ", Zero)

        Answer = np.matmul(inv(Df), Zero)
        #print("Answer: ", Answer)
        s1=Answer[0][0]
        s2=Answer[1][0]

        #Calculate the loss vector
        rhs = np.zeros((3, 1), np.float32)
        rhs[0, 0] = h*r13
        rhs[1, 0] = h*r23
        rhs[2, 0] = h*r33

        lhs = np.zeros((3, 1), np.float32)
        lhs[0, 0] = s1*u1-s2*u2
        lhs[1, 0] = s1*v1-s2*v2
        lhs[2, 0] = s1-s2

        #print("loss vector: ", lhs-rhs)    
        diff = np.matmul(inv(rot), lhs)
        #print("diff vector: ", cv.transpose(diff))    
        #print("s1: ", s1, ", s2: ", s2)

    return s1, s2

def RecoverGPSCoord(cb_coord):
    ecef_coord = np.matmul(cb_to_ecef_transform[:,0:3], cb_coord) + cv.transpose(np.array([cb_to_ecef_transform[:, 3]]))
    #print("ecef_coord: ", cv.transpose(ecef_coord))

    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    lon, lat, alt = pyproj.transform(ecef, lla, ecef_coord[0][0], ecef_coord[1][0], ecef_coord[2][0], radians=False)
    return (lat, lon, alt)

calib_corners = None
calib_imgpt = None
axis = np.float32([[0.5,0,0], [0,0.5,0], [0,0,-0.5]]).reshape(-1,3)
frame_num=0
hog = cv.HOGDescriptor()
hog.setSVMDetector( cv.HOGDescriptor_getDefaultPeopleDetector() )

fs_read = cv2.FileStorage("cb_to_ecef.yml", cv2.FILE_STORAGE_READ)
cb_to_ecef_transform = fs_read.getNode("transform").mat()
print("cb_to_ecef_transform: ", cb_to_ecef_transform)
fs_read.release()   

video_writer=None
now = datetime.datetime.now()
name = "output" + str(now.strftime("%Y-%m-%d-%H-%M-%S")) + ".avi"
if IsSaveToVideo:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(name, fourcc, 8, (1280, 720))
    print("Saving the frames to video output -> ", name)

while(cap.isOpened()):
    frame_num = frame_num+1
    start = time.time()
    ret, frame = cap.read()
    if ret == False:
        break
    elif IsSaveToVideo:
        video_writer.write(frame)

    #print("cap.read() took {} seconds.".format(time.time() - start))
    start = time.time()
    #cv2.putText(frame, str(datetime.datetime.now()), (210, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, 2)
    #time.sleep(0.055)

    if IsVanishingCalibration:
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
        #cv.imshow('frame',resized_frame)

        # f = plt.figure()
        # f.add_subplot(1,2, 1)
        # plt.imshow(resized_frame)
        # f.add_subplot(1,2, 2)
        # plt.imshow(resized_fgmask)
        # plt.show(block=True)

    largest_cb_square_dist=0
    if frame.any() and not FinishCalibration:
        fitting_error=[]
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #frame_gray[frame_bw==0]=0
        extrinsics = None
        CanStillFound = True
        DetectedChessBoardnum=0
        while CanStillFound:
            thrhd_value = 240
            bw_ret,frame_bw = cv.threshold(frame_gray,thrhd_value,255,cv.THRESH_BINARY)
            #frame_multi=np.zeros((720,1280),dtype="uint8")
            frame_bw = cv.morphologyEx(frame_bw, cv.MORPH_OPEN, kernel)
            frame_bw = cv.dilate(frame_bw,kernel3,iterations = 1)
            frame_multi = cv2.bitwise_and(frame_gray, frame_gray, mask=frame_bw)            
            cv.imwrite("zzzz_frame_gray.png", frame_gray)
            cv.imwrite("zzzz_frame_multi.png", frame_multi)
            cv.imwrite("zzzz_frame_bw.png", frame_bw)
            found, corners = cv.findChessboardCorners(frame_multi, pattern_size,  flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)
            if found:
                DetectedChessBoardnum = DetectedChessBoardnum + 1
                obj_points = []
                img_points = []   
                #print("corners:", corners)
                
                cv.cornerSubPix(frame_gray, corners, (5, 5), (-1, -1), term)
                #print("corners:", corners)                
                cb_square_dist = (corners[0][0][0]-corners[1][0][0])*(corners[0][0][0]-corners[1][0][0])+(corners[0][0][1]-corners[1][0][1])*(corners[0][0][1]-corners[1][0][1])
                print("cb_square_dist:", cb_square_dist)
                chessboards = [(corners.reshape(-1, 2), pattern_points)]

                chessboards = [x for x in chessboards if x is not None]
                for (corners, pattern_points) in chessboards:
                    img_points.append(corners)
                    obj_points.append(pattern_points)

                # calculate camera distortion
                h, w = frame_gray.shape[:2]  # TODO: use imquery call to retrieve results
                #rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), cameraMatrix=camera_matrix_manual, distCoeffs=dist_coefs_manual, flags=cv.CALIB_USE_INTRINSIC_GUESS+ cv.CALIB_FIX_K1+ cv.CALIB_FIX_K2+ cv.CALIB_FIX_K3+ cv.CALIB_FIX_K4+ cv.CALIB_FIX_K5)
                #print(img_points[0][0])
                returnval, rvecs, tvecs = cv.solvePnP(np.array(obj_points), np.array(img_points),camera_matrix_manual, dist_coefs_manual )

                if cb_square_dist > largest_cb_square_dist:
                    largest_cb_square_dist = cb_square_dist
                    ref_rvec = rvecs
                    ref_tvec = tvecs
                    #print(axis)
                    imgpts2, jac2 = cv.projectPoints(axis, rvecs, tvecs, camera_matrix_manual, dist_coefs_manual)
                    calib_corners = corners
                    calib_imgpt = imgpts2
                    imgpts3, jac3 = cv.projectPoints(grid, rvecs, tvecs, camera_matrix_manual, dist_coefs_manual)
                    ptgrid = imgpts3
                    #frame = draw(frame,calib_corners,calib_imgpt)

                OtherCBOrigin2D.append(corners[0])

                imgpts, jac = cv.projectPoints(np.array(obj_points), rvecs, tvecs, camera_matrix_manual, dist_coefs_manual)
                #print(imgpts[0][0])

                totalfittingerror=0
                for zz in range(len(imgpts)):
                    totalfittingerror = totalfittingerror + math.sqrt(math.pow(imgpts[zz][0][0]-img_points[0][zz][0], 2)+math.pow(imgpts[zz][0][1]-img_points[0][zz][1], 2))
                fitting_error.append(totalfittingerror)
                #print("totalfittingerror: ", totalfittingerror)



                #print("\nRMS:", rms)
                #print("camera matrix:\n", camera_matrix)
                #print("distortion coefficients: ", dist_coefs.ravel())     

                # brings the calibration pattern from the model coordinate space (in which object points are specified)
                # to the world coordinate space, that is, a real position of the calibration pattern
                # from chessboard (0, 0, 0) to 
                print("rotation: ",  [x* 180.0 / math.pi for x in rvecs])
                print("translation: ", cv.transpose(tvecs))     

                ext = cv.hconcat([np.array(cv.transpose(rvecs)), np.array(cv.transpose(tvecs))])
                print("ext: ", ext)

                if DetectedChessBoardnum == 1:
                    extrinsics = ext
                else:
                    extrinsics = np.vstack((extrinsics, ext))

                cv.drawChessboardCorners(frame, pattern_size, corners, found)

                #mask out the current chessboard
                x,y,w,h = cv.boundingRect(corners)
                margin=30
                cv.rectangle(frame_gray,(x-margin,y-margin),(x+w+margin,y+h+margin),255,-1)
                cv.putText(frame, str(DetectedChessBoardnum), (x, y), cv.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 5) 
                # cv.imshow('masked chessboard', frame_gray)
                # cv.waitKey(0)                
            
            else:
                print("Cannot find any chessboard! break!")
                CanStillFound = False
                break



        if DetectedChessBoardnum > 0:
            print("Calibration result is: ", ref_rvec, ref_tvec)
            board_width = 5
            board_height = 4
            square_size = 0.064

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_aspect("equal")

            cam_width = 0.064*6
            cam_height = 0.048*6
            scale_focal = 300
            min_values, max_values = draw_camera_boards(ax, camera_matrix_manual.copy(), cam_width, cam_height,
                                                        scale_focal, extrinsics, board_width,
                                                        board_height, square_size, False)

            X_min = min_values[0]
            X_max = max_values[0]
            Y_min = min_values[1]
            Y_max = max_values[1]
            Z_min = min_values[2]
            Z_max = max_values[2]
            max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

            mid_x = (X_max+X_min) * 0.5
            mid_y = (Y_max+Y_min) * 0.5
            mid_z = (Z_max+Z_min) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_zlabel('-y')
            ax.set_title('Extrinsic Parameters Visualization')

            for item in fitting_error:
                print("fitting error: ", item)
      
            FinishCalibration = True
            # cv.imshow('chessboard corners', frame)
            # cv.waitKey(0)   
            # plt.show()       

    ratio = 2.0
    ratioint = int(ratio)
    pick_inregion=[]
    margin=15
    if frame.any() and FinishCalibration:
        starts = time.time()
        frame_display = frame.copy()
        frame_display = draw_axis_and_ptgrid(frame_display,calib_corners,calib_imgpt, ptgrid)

        if IsDetectPeople:
            resized_frame = cv.resize(frame, (0,0), fx=(1/ratio), fy=(1/ratio)) 

            start = time.time()
            rects, weight = hog.detectMultiScale(resized_frame, winStride=(8, 8), padding=(32,32), scale=1.05)
            #print("hog.detectMultiScale took {} seconds.".format(time.time() - start))
            
            found_filtered = []
            # kill bb that has low weight
            for qz in range(len(rects)):
                if weight[qz][0] > hog_threshold:
                    found_filtered.append(rects[qz])

            # found_filtered = []
            # for ri, r in enumerate(rect):
            #     for qi, q in enumerate(rect):
            #         if ri != qi and inside(r, q):
            #             break
            #     else:
            #         found_filtered.append(r)
            # draw_detections(resized_frame, found_filtered, 3)
            found_filtered = np.array([[x, y, x + w, y + h] for (x, y, w, h) in found_filtered])
            pick = non_max_suppression(found_filtered, probs=None, overlapThresh=0.3)

            # draw the final bounding boxes
            for (xA, yA, xB, yB) in pick:
                ground_center = ((xA+xB)*0.5*ratio, max(yA, yB)*ratio)
                dist = cv2.pointPolygonTest(regionpts,ground_center,True)
                if not dist < -1 :
                    marginpt = ((xA +margin)*ratioint, (yA+margin)*ratioint, (xB-margin)*ratioint, (yB-margin)*ratioint)
                    pick_inregion.append(marginpt)
                    cv.rectangle(frame_display, (marginpt[0], marginpt[1]), (marginpt[2], marginpt[3]), (0, 255, 0), 2)        

        #bigger_frame = cv.resize(resized_frame, (0,0), fx=ratio, fy=ratio) 
        #cv.imshow('pedestrian detection', bigger_frame)        

        Size_of_w=600
        physical_size=30
        Lmap = GetWindowWithAxis(Size_of_w, physical_size)
        Lmap_localized, bigger_frame_reproject = PrintLocalization(Lmap, frame_display, pick_inregion, 1.0, Size_of_w, physical_size, camera_matrix_manual, dist_coefs_manual, ref_rvec, ref_tvec, calib_corners)

        #print("whole loop took {} seconds.".format(time.time() - starts))

        cv.imshow('pedestrian detection', bigger_frame_reproject)        
        cv.imshow('localization', Lmap_localized)      

        #Pending for debug
        if IsEachFrameDebug and len(rects)>0:
            cv.waitKey(5000)

    #cv.imshow('fgmask',resized_fgmask)
    #cv.imshow('axis',resized_fitline)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if IsSaveToVideo:
    video_writer.release()
cv.destroyAllWindows()

# print("Collected over {} major axis of pedestrian blob, start calculation...".format(line_db_need_to_collect))

# #Allocate the voting space
# voting_space={}
# for line1 in line_db:
#     for line2 in line_db:
#         ret = find_two_line_intersection(line1, line2)
#         if ret[0] == True:
#             int_coord = (int(ret[1]), int(ret[2]))
#             if int_coord in voting_space:
#                 voting_space[int_coord] = voting_space[int_coord]+1
#             else:
#                 voting_space[int_coord] = 1

# if voting_space:
#     vote_coord = max(voting_space.items(), key=operator.itemgetter(1))[0]
#     print("Finish voting the pixel level vanishing point, which is {}, have {} vote".format(vote_coord, voting_space[vote_coord]))


#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

import sys
sys.path.insert(0, "../../ncapi2_shim")
import mvnc_simple_api as mvnc
#from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
import cv2 as cv
import time
import csv
import os
import sys
from sys import argv

import numpy as np
import datetime
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
square_size = 0.06395
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
FinishCalibration = False
ref_rvec=None
ref_tvec=None
calib_corners = None
calib_imgpt = None
ptgrid = None
axis = np.float32([[0.5,0,0], [0,0.5,0], [0,0,-0.5]]).reshape(-1,3)
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

# name of the opencv window
cv_window_name = "SSD Mobilenet" 

# labels AKA classes.  The class IDs returned
# are the indices into this list
labels = ('background','aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus',
            'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

# only accept classifications with 1 in the class id index.
# default is to accept all object clasifications.
# for example if object_classifications_mask[1] == 0 then
#    will ignore aeroplanes
# object_classifications_mask = [1, 1, 1, 1, 1, 1, 1,
#                                1, 1, 1, 1, 1, 1, 1,
#                                1, 1, 1, 1, 1, 1, 1]

object_classifications_mask = [1, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0,
                               0, 1, 0, 0, 0, 0, 0]                                   

# the ssd mobilenet image width and height
NETWORK_IMAGE_WIDTH = 300
NETWORK_IMAGE_HEIGHT = 300

# the minimal score for a box to be shown
DEFAULT_INIT_MIN_SCORE = 20
min_score_percent = DEFAULT_INIT_MIN_SCORE

# the resize_window arg will modify these if its specified on the commandline
resize_output = False
resize_output_width = 0
resize_output_height = 0



# read video files from this directory
input_video_path = '.'



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

def GetBottomPtScale(head_pt, bottom_pt, rot, ref_tvec, method=0):
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
        print("s2: ", s2)

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
        print("Df: ", Df)

        Zero = np.zeros((2, 1), np.float32)
        Zero[0, 0] = 2*h*(u1*r13+v1*r23+r33)
        Zero[1, 0] = -2*h*(u1*r13+v2*r23+r33)
        print("Zero: ", Zero)

        Answer = inv(Df)*Zero
        print("Answer: ", Answer)
        s1=Answer[0][0]
        s2=Answer[1][0]
        print("s1: ", s1, ", s2: ", s2)

    return s2

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

        print("bottom_pt: ", bottom_pt)
        bp = np.zeros((1, 2), np.float32)
        bp[0][0] = bottom_pt[0]
        bp[0][1] = bottom_pt[1]
        #print("calib_corners: ", calib_corners[0])
        dst = cv.undistortPoints(bp.reshape(-1,1,2).astype(np.float32), camera_matrix_manual, dist_coefs_manual)
        bottom_pt = dst[0][0]
        # according to the formula: s1=1.7*r33+s2, s2=1.7*(r23-v1*r33)/(v1-v2)
        #s2 = 1.7*(rot[1][2]-head_pt[1]*rot[2][2])/(head_pt[1]-bottom_pt[1])
        print("bottom_pt after undistort: ", bottom_pt)
        #bottom_pt_center_normalized = (bottom_pt[0]-camera_matrix_manual[0][2], bottom_pt[1]-camera_matrix_manual[1][2])
        #print("bottom_pt_center_normalized: ", bottom_pt_center_normalized)        
        #s1 = 1.7*rot[2][2]+s2

        #print("ref_tvec: ", ref_tvec[0][0], ref_tvec[1][0], ref_tvec[2][0])
        AlgMethod=1  # 0 : assume z=0, 1: assume height=1.7
        s2 = GetBottomPtScale(head_pt, bottom_pt, rot, ref_tvec, AlgMethod)

        img_pt = np.array([[bottom_pt[0]], [bottom_pt[1]], [1]])
        #print("zz: ", s2*img_pt-ref_tvec)
        #print("tra: ", cv.transpose(rot))
        result2 = np.matmul(cv.transpose(rot), s2*img_pt-ref_tvec)
        print("result: ",cv.transpose(result2))

        center = (int(Size_of_w*0.5-result2[1][0]*px_of_meter), int(result2[0][0]*px_of_meter+Size_of_w*0.5))     
        print("world XY: ", result2[0][0], result2[1][0])    
        print("window XY: ", center[0], center[1])
        #print("result: ", result)
        pchar = "[" + str(round_to_1(result2[0][0])) + ", " + str(round_to_1(result2[1][0])) + "]"

        Lmap_localized = cv.circle(Lmap_localized, center, 5, (255,0,0), thickness=3, lineType=8, shift=0) 
        cv.putText(Lmap_localized, pchar, center, cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1) 

        #Project the resulting 3d point onto 2d image again for comfirmation.
        imgpts, jac = cv.projectPoints(np.array([[result2[0][0], result2[1][0], result2[2][0]], [0.0, 0.0, 0.0]]), ref_rvec, ref_tvec, camera_matrix_manual, dist_coefs_manual)
        intpt =  (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
        print("intpt: ", intpt)

        if intpt[0]>0 and intpt[0]<bigger_frame_reproject.shape[1] and intpt[1]>0 and intpt[1]<bigger_frame_reproject.shape[0]:
            bigger_frame_reproject = cv.circle(bigger_frame_reproject, intpt, 5, (255,0,0), thickness=3, lineType=8, shift=0) 
            cv.putText(bigger_frame_reproject, pchar, (intpt[0], intpt[1]+30), cv.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 2)         

    return Lmap_localized, bigger_frame_reproject

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

# create a preprocessed image from the source image that complies to the
# network expectations and return it
def preprocess_image(source_image):
    resized_image = cv2.resize(source_image, (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT))
    
    # trasnform values from range 0-255 to range -1.0 - 1.0
    resized_image = resized_image - 127.5
    resized_image = resized_image * 0.007843
    return resized_image

# handles key presses by adjusting global thresholds etc.
# raw_key is the return value from cv2.waitkey
# returns False if program should end, or True if should continue
def handle_keys(raw_key):
    global min_score_percent
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False
    elif (ascii_code == ord('B')):
        min_score_percent += 5
        print('New minimum box percentage: ' + str(min_score_percent) + '%')
    elif (ascii_code == ord('b')):
        min_score_percent -= 5
        print('New minimum box percentage: ' + str(min_score_percent) + '%')

    return True


# overlays the boxes and labels onto the display image.
# display_image is the image on which to overlay the boxes/labels
# object_info is a list of 7 values as returned from the network
#     These 7 values describe the object found and they are:
#         0: image_id (always 0 for myriad)
#         1: class_id (this is an index into labels)
#         2: score (this is the probability for the class)
#         3: box left location within image as number between 0.0 and 1.0
#         4: box top location within image as number between 0.0 and 1.0
#         5: box right location within image as number between 0.0 and 1.0
#         6: box bottom location within image as number between 0.0 and 1.0
# returns None
def overlay_on_image(display_image, object_info):
    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = int(object_info[base_index + 1])
    if (class_id < 0):
        return

    if (object_classifications_mask[class_id] == 0):
        return

    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        return

    label_text = labels[class_id] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    scale_max = (100.0 - min_score_percent)
    scaled_prob = (percentage - min_score_percent)
    scale = scaled_prob / scale_max

    # draw the classification label string just above and to the left of the rectangle
    #label_background_color = (70, 120, 70)  # greyish green background for text
    label_background_color = (0, int(scale * 175), 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1] - 8
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

def draw_axis_and_ptgrid(img, corners, imgpts, ptgrid):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 3)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 3)

    # Also draw a point grid in the z-plane for debug purpose
    for pt in ptgrid:
        img = cv.circle(img, tuple(pt.ravel()), 4, (0,255,0), thickness=2, lineType=8, shift=0) 

    return img

#return False if found invalid args or True if processed ok
def handle_args():
    global resize_output, resize_output_width, resize_output_height, min_score_percent, object_classifications_mask
    for an_arg in argv:
        if (an_arg == argv[0]):
            continue

        elif (str(an_arg).lower() == 'help'):
            return False

        elif (str(an_arg).lower().startswith('exclude_classes=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                exclude_list = str(val).split(',')
                for exclude_id_str in exclude_list:
                    exclude_id = int(exclude_id_str)
                    if (exclude_id < 0 or exclude_id>len(labels)):
                        print("invalid exclude_classes= parameter")
                        return False
                    print("Excluding class ID " + str(exclude_id) + " : " + labels[exclude_id])
                    object_classifications_mask[int(exclude_id)] = 0
            except:
                print('Error with exclude_classes argument. ')
                return False;

        elif (str(an_arg).lower().startswith('init_min_score=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                init_min_score_str = val
                init_min_score = int(init_min_score_str)
                if (init_min_score < 0 or init_min_score > 100):
                    print('Error with init_min_score argument.  It must be between 0-100')
                    return False
                min_score_percent = init_min_score
                print ('Initial Minimum Score: ' + str(min_score_percent) + ' %')
            except:
                print('Error with init_min_score argument.  It must be between 0-100')
                return False;

        elif (str(an_arg).lower().startswith('resize_window=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                width_height = str(val).split('x', 1)
                resize_output_width = int(width_height[0])
                resize_output_height = int(width_height[1])
                resize_output = True
                print ('GUI window resize now on: \n  width = ' +
                       str(resize_output_width) +
                       '\n  height = ' + str(resize_output_height))
            except:
                print('Error with resize_window argument: "' + an_arg + '"')
                return False
        else:
            return False

    return True


# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def run_inference(image_to_classify, ssd_mobilenet_graph):

    regionpts = numpy.array([[361,195],[414,195],[558,691],[761,685]], numpy.int32)
    # preprocess the image to meet nework expectations
    resized_image = preprocess_image(image_to_classify)

    # Send the image to the NCS as 16 bit floats
    ssd_mobilenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # Get the result from the NCS
    output, userobj = ssd_mobilenet_graph.GetResult()

    #   a.	First fp16 value holds the number of valid detections = num_valid.
    #   b.	The next 6 values are unused.
    #   c.	The next (7 * num_valid) values contain the valid detections data
    #       Each group of 7 values will describe an object/box These 7 values in order.
    #       The values are:
    #         0: image_id (always 0)
    #         1: class_id (this is an index into labels)
    #         2: score (this is the probability for the class)
    #         3: box left location within image as number between 0.0 and 1.0
    #         4: box top location within image as number between 0.0 and 1.0
    #         5: box right location within image as number between 0.0 and 1.0
    #         6: box bottom location within image as number between 0.0 and 1.0

    # number of boxes returned
    num_valid_boxes = int(output[0])
    found_filtered = []
    probs=[]
    for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not numpy.isfinite(output[base_index]) or
                    not numpy.isfinite(output[base_index + 1]) or
                    not numpy.isfinite(output[base_index + 2]) or
                    not numpy.isfinite(output[base_index + 3]) or
                    not numpy.isfinite(output[base_index + 4]) or
                    not numpy.isfinite(output[base_index + 5]) or
                    not numpy.isfinite(output[base_index + 6])):
                # boxes with non finite (inf, nan, etc) numbers must be ignored
                continue

            x1 = max(int(output[base_index + 3] * image_to_classify.shape[1]), 0)
            y1 = max(int(output[base_index + 4] * image_to_classify.shape[0]), 0)
            x2 = min(int(output[base_index + 5] * image_to_classify.shape[1]), image_to_classify.shape[1]-1)
            y2 = min((output[base_index + 6] * image_to_classify.shape[0]), image_to_classify.shape[0]-1)

            class_id = int(output[base_index + 1])
            percentage = int(output[base_index + 2] * 100)
            ground_center = ((x1+x2)*0.5, max(y1, y2))
            dist = cv2.pointPolygonTest(regionpts,ground_center,True)
            if class_id >= 0 and object_classifications_mask[class_id] != 0 and percentage > min_score_percent and not dist < -1:
                found_filtered.append([x1, y1, x2, y2])
                probs.append(output[base_index + 2])

                # overlay boxes and labels on to the image
                overlay_on_image(image_to_classify, output[base_index:base_index + 7])

    # display text to let user know how to quit
    cv2.rectangle(image_to_classify,(0, 0),(100, 15), (128, 128, 128), -1)
    cv2.putText(image_to_classify, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    ratio = 1.0
    if image_to_classify.any() and FinishCalibration:
        frame = draw_axis_and_ptgrid(image_to_classify,calib_corners,calib_imgpt, ptgrid)
        pick = non_max_suppression(np.array(found_filtered), probs, overlapThresh=0.3)

        # draw the final bounding boxes
        #pick_masked=[]
        #regionpts = numpy.array([[306,164],[352,164],[508,676],[770,676]], numpy.int32)
        for (xA, yA, xB, yB) in pick:
            # ground_center = ((xA+xB)*0.5, max(yA, yB))
            # dist = cv2.pointPolygonTest(regionpts,ground_center,True)
            # if dist>0:
            cv.rectangle(image_to_classify, (xA, yA), (xB, yB), (0, 255, 0), 2)     
                #pick_masked.append((xA, yA, xB, yB))   

        Size_of_w=600
        physical_size=30
        Lmap = GetWindowWithAxis(Size_of_w, physical_size)
        Lmap_localized, bigger_frame_reproject = PrintLocalization(Lmap, image_to_classify, pick, ratio, Size_of_w, physical_size, camera_matrix_manual, dist_coefs_manual, ref_rvec, ref_tvec, calib_corners)
        cv.imshow('pedestrian detection', bigger_frame_reproject)        
        cv.imshow('localization', Lmap_localized)     

# prints usage information
def print_usage():
    print('\nusage: ')
    print('python3 run_video.py [help][resize_window=<width>x<height>]')
    print('')
    print('options:')
    print('  help - prints this message')
    print('  resize_window - resizes the GUI window to specified dimensions')
    print('                  must be formated similar to resize_window=1280x720')
    print('                  Default isto not resize, use size of video frames.')
    print('  init_min_score - set the minimum score for a box to be recognized')
    print('                  must be a number between 0 and 100 inclusive.')
    print('                  Default is: ' + str(DEFAULT_INIT_MIN_SCORE))

    print('  exclude - comma separated list of object class IDs to exclude from following:')
    index = 0
    for oneLabel in labels:
        print("                 class ID " + str(index) + ": " + oneLabel)
        index += 1
    print('            must be a number between 0 and ' + str(len(labels)) + ' inclusive.')
    print('            Default is to exclude none.')

    print('')
    print('Example: ')
    print('python3 run_video.py resize_window=1920x1080 init_min_score=50 exclude_classes=5,11')


# This function is called from the entry point to do
# all the work.
def main():
    global resize_output, resize_output_width, resize_output_height, FinishCalibration, pattern_size, pattern_points, camera_matrix_manual, dist_coefs_manual, ref_rvec, ref_tvec, axis, calib_corners, calib_imgpt, ptgrid
    if (not handle_args()):
        print_usage()
        return 1

    # configure the NCS
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

    # Get a list of ALL the sticks that are plugged in
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])

    # Open the NCS
    device.OpenDevice()

    graph_filename = 'graph'

    # Load graph file to memory buffer
    with open(graph_filename, mode='rb') as f:
        graph_data = f.read()

    # allocate the Graph instance from NCAPI by passing the memory buffer
    ssd_mobilenet_graph = device.AllocateGraph(graph_data)

    # get list of all the .mp4 files in the image directory
    input_video_filename_list = []
    input_video_filename_list.append('rtsp://admin:h0940232@172.18.9.100/Streaming/Channels/1')
    #input_video_filename_list.append('sample.MOV')


    # input_video_filename_list = os.listdir(input_video_path)
    # input_video_filename_list = [i for i in input_video_filename_list if i.endswith('.mp4')]

    # if (len(input_video_filename_list) < 1):
    #     # no images to show
    #     print('No video (.mp4) files found')
    #     return 1

    cv2.namedWindow(cv_window_name)
    cv2.moveWindow(cv_window_name, 10,  10)

    exit_app = False
    while (True):
        for input_video_file in input_video_filename_list :

            cap = cv2.VideoCapture(input_video_file)

            actual_frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print ('actual video resolution: ' + str(actual_frame_width) + ' x ' + str(actual_frame_height))

            if ((cap == None) or (not cap.isOpened())):
                print ('Could not open video device.  Make sure file exists:')
                print ('file name:' + input_video_file)
                print ('Also, if you installed python opencv via pip or pip3 you')
                print ('need to uninstall it and install from source with -D WITH_V4L=ON')
                print ('Use the provided script: install-opencv-from_source.sh')
                exit_app = True
                break

            frame_count = 0
            start_time = time.time()
            end_time = start_time

            while(True):
                ret, frame = cap.read()
                #display_image=cv2.rotate(display_image_raw, cv2.ROTATE_90_CLOCKWISE)

                if (not ret):
                    end_time = time.time()
                    print("No image from from video device, exiting")
                    break

                # check if the window is visible, this means the user hasn't closed
                # the window via the X button
                prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
                if (prop_val < 0.0):
                    end_time = time.time()
                    exit_app = True
                    break

                run_inference(frame, ssd_mobilenet_graph)

                if frame.any() and not FinishCalibration:
                    fitting_error=[]
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    extrinsics = None
                    CanStillFound = True
                    DetectedChessBoardnum=0
                    while CanStillFound:
                        #cv.imwrite("zzzz.png", frame_gray)
                        found, corners = cv.findChessboardCorners(frame_gray, pattern_size,  flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)
                        if found:
                            DetectedChessBoardnum = DetectedChessBoardnum + 1
                            obj_points = []
                            img_points = []   
                            #print("corners:", corners)
                            
                            cv.cornerSubPix(frame_gray, corners, (5, 5), (-1, -1), term)
                            #print("corners:", corners)
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

                            #if img_points[0][0][0] > 640:
                            ref_rvec = rvecs
                            ref_tvec = tvecs
                            print("Calibration result is: ", ref_rvec, ref_tvec)
                            print(axis)
                            imgpts2, jac2 = cv.projectPoints(axis, rvecs, tvecs, camera_matrix_manual, dist_coefs_manual)
                            calib_corners = corners
                            calib_imgpt = imgpts2
                            imgpts3, jac3 = cv.projectPoints(grid, rvecs, tvecs, camera_matrix_manual, dist_coefs_manual)
                            ptgrid = imgpts3
                            #frame = draw(frame,calib_corners,calib_imgpt)

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
                            cv.rectangle(frame_gray,(x-10,y-10),(x+w+10,y+h+10),255,-1)
                            cv.putText(frame, str(DetectedChessBoardnum), (x, y), cv.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 5) 
                            # cv.imshow('masked chessboard', frame_gray)
                            # cv.waitKey(0)                
                        
                        else:
                            print("Cannot find any chessboard! break!")
                            CanStillFound = False
                            break



                    if DetectedChessBoardnum > 0:
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

                if (resize_output):
                    frame = cv2.resize(frame,
                                               (resize_output_width, resize_output_height),
                                               cv2.INTER_LINEAR)
                #cv2.imshow(cv_window_name, frame)

                raw_key = cv2.waitKey(1)
                if (raw_key != -1):
                    if (handle_keys(raw_key) == False):
                        end_time = time.time()
                        exit_app = True
                        break
                frame_count += 1

            frames_per_second = frame_count / (end_time - start_time)
            print('Frames per Second: ' + str(frames_per_second))

            cap.release()

            if (exit_app):
                break;

        if (exit_app):
            break

    # Clean up the graph and the device
    ssd_mobilenet_graph.DeallocateGraph()
    device.CloseDevice()


    cv2.destroyAllWindows()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())

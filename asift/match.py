"""
This SOFTWARE PRODUCT is provided by ASTRI "as is" and "with all faults." 
ASTRI makes no representations or warranties of any kind concerning the safety,
suitability, lack of viruses, inaccuracies, typographical errors, or other harmful
components of this SOFTWARE PRODUCT. There are inherent dangers in the use of any software,
and you are solely responsible for determining whether this SOFTWARE PRODUCT is
compatible with your equipment and other software installed on your equipment. 
You are also solely responsible for the protection of your equipment and backup of your data,
and ASTRI will not be liable for any damages you may suffer in connection with using,
modifying, or distributing this SOFTWARE PRODUCT.

Copyright (C) 2020, ASTRI, all rights reserved.
Third party copyrights are property of their respective owners.

[Camera automatic calibration module]

Module to calibrate the homography transformation from image plane coordinate to ground plane gps coordinate.
"""

#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import cv2

# built-in modules
import itertools as it
from multiprocessing.pool import ThreadPool

# local modules
from common import Timer
import os
from matplotlib import pyplot as plt
import time, datetime
import math
import pyproj
from osgeo import gdal
import sys
import getopt

import logging
import logging.handlers
from logging.handlers import RotatingFileHandler

# for marker matching
from sklearn.utils.linear_assignment_ import linear_assignment
import random

import subprocess
def GetLatLon(line):
    coords = line.split(') (')[1]
    coords = coords[:-1]
    LonStr, LatStr = coords.split(',')
    # Longitude
    LonStr = LonStr.split('d')    # Get the degrees, and the rest
    LonD = int(LonStr[0])
    LonStr = LonStr[1].split('\'')# Get the arc-m, and the rest
    LonM = int(LonStr[0])
    LonStr = LonStr[1].split('"') # Get the arc-s, and the rest
    LonS = float(LonStr[0])
    Lon = LonD + LonM/60. + LonS/3600.
    if LonStr[1] in ['W', 'w']:
        Lon = -1*Lon
    # Same for Latitude
    LatStr = LatStr.split('d')
    LatD = int(LatStr[0])
    LatStr = LatStr[1].split('\'')
    LatM = int(LatStr[0])
    LatStr = LatStr[1].split('"')
    LatS = float(LatStr[0])
    Lat = LatD + LatM/60. + LatS/3600.
    if LatStr[1] in ['S', 's']:
        Lat = -1*Lat
    return Lat, Lon
    
def GetCornerCoordinates(FileName):
    GdalInfo = subprocess.check_output('gdalinfo {}'.format(FileName), shell=True)
    GdalInfo = GdalInfo.decode('utf-8')

    decoded_lines = GdalInfo.split('\n')
    CornerLats, CornerLons = np.zeros(5), np.zeros(5)
    GotUL, GotUR, GotLL, GotLR, GotC = False, False, False, False, False
    
    for line in decoded_lines:
        if line[:10] == 'Upper Left':
            CornerLats[0], CornerLons[0] = GetLatLon(line)
            GotUL = True
        if line[:10] == 'Lower Left':
            CornerLats[1], CornerLons[1] = GetLatLon(line)
            GotLL = True
        if line[:11] == 'Upper Right':
            CornerLats[2], CornerLons[2] = GetLatLon(line)
            GotUR = True
        if line[:11] == 'Lower Right':
            CornerLats[3], CornerLons[3] = GetLatLon(line)
            GotLR = True
        if line[:6] == 'Center':
            CornerLats[4], CornerLons[4] = GetLatLon(line)
            GotC = True
        if GotUL and GotUR and GotLL and GotLR and GotC:
            break
    return CornerLats, CornerLons 

def simplify_contour(contour, n_corners=4, max_iter=100):
    """
    Binary searches best `epsilon` value to force contour approximation contain exactly `n_corners` points.

    Note:
        N/A

    Args:
        contour (nparray): input contour, opencv contour object
        n_corners (int): Number of corners (points) the contour must contain.
        max_iter (int): maximum iteration to try to force contour approximation contain exactly `n_corners` points.

    Returns:
        contour (nparray): Simplified contour in successful case. Otherwise returns initial contour.

    Raises:
        N/A
        
    """    
    n_iter = 0
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

def namestr(obj, namespace):
    """
    Return the name of the variable, for debug message usege.

    Note:
        N/A

    Args:
        obj: variable
        namespace: python namespace

    Returns:
        variable_name (str)

    Raises:
        N/A
        
    """   
    return [name for name in namespace if namespace[name] is obj]

def linelength(line):
    """
    Return the length of the input line

    Note:
        N/A

    Args:
        line (list): [x1, y1, x2, y2] representing the 2 end point of the line

    Returns:
        length (double)

    Raises:
        N/A
        
    """   
    return math.sqrt(math.pow((line[2]-line[0]),2)+math.pow((line[3]-line[1]),2))

def slope_of_line_in_angle(line):
    """
    Return the slope of the input line

    Note:
        N/A

    Args:
        line (list): [x1, y1, x2, y2] representing the 2 end point of the line

    Returns:
        slope (double)

    Raises:
        N/A
        
    """   
    return math.degrees(math.atan2(1.0*line[3]-1.0*line[1], (1.0*line[2]-1.0*line[0])))

def checksameline(line1, line2, threshold=2.0):
    """
    Function to check whether 2 input lines are essentially the same

    Note:
        N/A

    Args:
        line1 (list): [x1, y1, x2, y2] representing the 2 end point of the line
        line2 (list): [x1, y1, x2, y2] representing the 2 end point of the line
        threshold (double): distance between the end points

    Returns:
        True/False (Boolean)

    Raises:
        N/A
        
    """   
    dist1 = math.sqrt(math.pow((line1[0]-line2[0]),2)+math.pow((line1[1]-line2[1]),2))
    dist2 = math.sqrt(math.pow((line1[0]-line2[2]),2)+math.pow((line1[1]-line2[3]),2))
    dist3 = math.sqrt(math.pow((line1[2]-line2[0]),2)+math.pow((line1[3]-line2[1]),2))
    dist4 = math.sqrt(math.pow((line1[2]-line2[2]),2)+math.pow((line1[3]-line2[3]),2))

    if min(dist1, dist2) < threshold and min(dist3, dist4) < threshold:
        return True
    else:
        return False

def intersect(line1, line2): 
    pt1 = (line1[0], line1[1])
    pt2 = (line1[2], line1[3])
    ptA = (line2[0], line2[1])
    ptB = (line2[2], line2[3])

    """
    This function returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

    Note:
        N/A

    Args:
        line1 (list): [x1, y1, x2, y2] representing the 2 end point of the line
        line2 (list): [x1, y1, x2, y2] representing the 2 end point of the line

    Returns:
        (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment

    Raises:
        N/A
        
    """   
    DET_TOLERANCE = 1e-10

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


def lines_merging(lines, line_min_length=5.0, lines_angle_diff_in_degree=5.0, extension_ratio=2.0, maxgap=50.0):
    """
    This function perform line merging of the input lines

    Note:
        N/A

    Args:
        line (list): [x1, y1, x2, y2] representing the 2 end point of the line
        line_min_length (double): line shorter than this threshold will not be considered
        lines_angle_diff_in_degree (double): 2 lines angle difference smaller than this will be merged
        extension_ratio (double): How long the original lines need to be extended to reach their intersection
        maxgap (double): Distance threshold for how far the end point of the lines to their intersection point

    Returns:
        line (list): A list of merged lines

    Raises:
        N/A
        
    """  

    new_extended_lines=[]
    lines_with_info=[]

    # First, filter lines which is too short
    for line in lines:
        length=linelength(line[0])

        if length>line_min_length:
            lines_with_info.append([line[0], slope_of_line_in_angle(line[0]), length])

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

        # also append the original segment if no merging happen to him
        if not CanMerge:
            new_extended_lines.append(line1)

    return new_extended_lines

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

def affine_skew(tilt, phi, img, mask=None):
    """
    function to perform affine distortion

    Note:
        N/A

    Args:
        tile (double): defined tile angle
        phi (double): defined phi angle
        img (nparray): input image
        mask (nparray): input mask

    Returns:
        img (nparray): image after affine distortion (skew_img)
        mask (nparray): mask after affine distortion (skew_mask)
        Ai (nparray): affine transform matrix from skew_img to img 

    Raises:
        N/A
    
    """    
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv.warpAffine(img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
    Ai = cv.invertAffineTransform(A)
    return img, mask, Ai

def affine_detect(detector, img, mask=None, pool=None):
    """
    function to detect keypoints 

    Note:
        N/A

    Args:
        detector (opencv defined type): keypoint detector
        img (nparray): input image
        mask (nparray): input mask
        pool : number of threads to be used

    Returns:
        keypoints (opencv defined type): key points of input image
        descrs (opencv defined type): descriptors of input image key points

    Raises:
        N/A
    
    """    
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img, mask)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)

        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        keypoints.extend(k)
        descrs.extend(d)

    return keypoints, np.array(descrs)

def ConvertPx2GPS(x, y, scale_x, scale_y, img_cols, img_rows, min_x, min_y, max_x, max_y):

    """
    function to convert pixel coordinate to gps

    Note:
        N/A

    Args:
        x (int): column index of the image
        y (int): row index of the image
        scale_x (double): scale down factor in column direction
        scale_y (double): scale down factor in row direction
        img_cols (int): column size of orthophoto in original size
        img_rows (int): row size of orthophoto in original size
        min_x (double): min x value in gps coordinate
        min_y (double): min y value in gps coordinate
        max_x (double): max x value in gps coordinate
        max_y (double): max y value in gps coordinate

    Returns:
        gps (nparray): gps coordinate

    Raises:
        N/A
    
    """

    x = x/scale_x
    y = y/scale_y
    gps_x = max_x + (y/img_rows)*(min_x-max_x)
    gps_y = min_y + (x/img_cols)*(max_y-min_y)
    gps = [np.array([gps_x, gps_y])]

    return gps

def init_feature(name):

    """
    function to select feature detector and matcher

    Note:
        N/A

    Args:
        name (string): name of feature detector and matcher

    Returns:
        detector (N/A): feature detector
        matcher (N/A): feature matcher

    Raises:
        N/A
    
    """

    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv.xfeatures2d.SIFT_create()
        norm = cv.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv.xfeatures2d.SURF_create(800)
        norm = cv.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv.ORB_create(400)
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv.AKAZE_create()
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv.BRISK_create()
        norm = cv.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv.BFMatcher(norm)
    return detector, matcher


# def check_inside(roi, pt):
#     if pt[0] >= 0 and pt[0] < roi[1] and pt[1] >= 0 and pt[1] < roi[0]:
#         return True
#     else: 
#         return False

# def mid(pt1, pt2):
#     return (int((pt1[0]+pt2[0])/2), int((pt1[1]+pt2[1])/2))

# def Drawline_maybe_outofboundary(img, pt1, pt2, color, thickness):
#     check_pt1 = check_inside(img.shape, pt1)
#     check_pt2 = check_inside(img.shape, pt2)

#     if check_pt1 and check_pt2:
#         cv.line(img, pt1, pt2, color, thickness)

#     elif check_pt1 and not check_pt2:
#         mid_pt = mid(pt1, pt2)
#         while not check_inside(img.shape, mid_pt):
#             mid_pt = mid(pt1, mid_pt)
#         cv.line(img, pt1, mid_pt, color, thickness)
    
#     elif not check_pt1 and check_pt2:
#         mid_pt = mid(pt1, pt2)
#         while not check_inside(img.shape, mid_pt):
#             mid_pt = mid(mid_pt, pt2)
#         cv.line(img, mid_pt, pt2, color, thickness)
    


def explore_match(img1, img2, kp_pairs, isOnlyKeyPt, pp1, pp2, H = None, logger = None, 
                output_dir = None, IsDumpPtPair = False, Num_of_additional_point_pair = 0, pp1_gps = None):

    """
    function to select feature detector and matcher

    Note:
        N/A

    Args:
        img1 (nparray): geotiff
        img2 (nparray): perspective image
        kp_pairs (opencv defined type): key point matching pairs 
        kp_pairs_inlier (opencv defined type): key point matching pairs filtered only inlier
        H (nparray): homography matrix from img2 to img1

    Returns:
        vis (nparray): image showing inlier matching pairs
        

    Raises:
        N/A
    
    """

    height_margin=0 # additional dark space at the bottom to allow the drawing of the reprojected corners
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2)+height_margin, w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
    vis_copy = vis.copy()

    # draw the lines in the boundingbox of perspective image
    #cv2.line(vis, (w1, 0), (w1+w2, 0), (255, 0, 0), 15)
    cv2.line(vis, (w1+w2, 0), (w1+w2, h2), (255, 0, 0), 15)
    cv2.line(vis, (w1+w2, h2), (w1, h2), (0, 0, 255), 10)
    cv2.line(vis, (w1, h2), (w1, 0), (255, 0, 0), 10)
    #cv.circle(vis ,(int(w1+w2*0.5), int(h2*0.5)), 13, (255, 0, 0), 3)

    # draw a red, blue and green cross the represent 3 points (from center to somewhere between center to Top mid pt) for visual verification
    cv2.line(vis, (int(w1+w2*0.5)-10, int(h2*0.5)-10), (int(w1+w2*0.5)+10, int(h2*0.5)+10), (0, 0, 255), 3)
    cv2.line(vis, (int(w1+w2*0.5)+10, int(h2*0.5)-10), (int(w1+w2*0.5)-10, int(h2*0.5)+10), (0, 0, 255), 3)

    cv2.line(vis, (int(w1+w2*0.5)-10, int(h2*0.25)-10), (int(w1+w2*0.5)+10, int(h2*0.25)+10), (255, 0, 0), 3)
    cv2.line(vis, (int(w1+w2*0.5)+10, int(h2*0.25)-10), (int(w1+w2*0.5)-10, int(h2*0.25)+10), (255, 0, 0), 3)

    cv2.line(vis, (int(w1+w2*0.5)-10, int(h2*0.125)-10), (int(w1+w2*0.5)+10, int(h2*0.125)+10), (0, 255, 0), 3)
    cv2.line(vis, (int(w1+w2*0.5)+10, int(h2*0.125)-10), (int(w1+w2*0.5)-10, int(h2*0.125)+10), (0, 255, 0), 3)

    # try to draw the range of view of perspective image into orthomoasic geotiff
    ratiostep = 0.025
    if H is not None:
        corners = np.float32([[w2, h2], [0, h2]])
        base = ratiostep
        while base < 1:
            corners = np.append(corners, np.float32([[w2, h2*base], [0, h2*base]]), axis=0)
            base = base + ratiostep

        corners = cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (0, 0)

        center = np.float32([[w2*0.5, h2*0.5], [w2*0.5, h2*0.25], [w2*0.5, h2*0.125]])
        center = cv.perspectiveTransform(center.reshape(1, -1, 2), H).reshape(-1, 2) + (0, 0)

        # Draw the bottom line as red
        cv.line(vis, (int(corners[0][0]), int(corners[0][1])), (int(corners[1][0]), int(corners[1][1])), (0, 0, 255), 5)

        # draw a red, blue and green cross the represent 3 points (from center to somewhere between center to Top mid pt) for visual verification
        cv2.line(vis, (int(center[0][0])-10, int(center[0][1])-10), (int(center[0][0])+10, int(center[0][1])+10), (0, 0, 255), 5)
        cv2.line(vis, (int(center[0][0])+10, int(center[0][1])-10), (int(center[0][0])-10, int(center[0][1])+10), (0, 0, 255), 5)        

        cv2.line(vis, (int(center[1][0])-10, int(center[1][1])-10), (int(center[1][0])+10, int(center[1][1])+10), (255, 0, 0), 5)
        cv2.line(vis, (int(center[1][0])+10, int(center[1][1])-10), (int(center[1][0])-10, int(center[1][1])+10), (255, 0, 0), 5) 

        cv2.line(vis, (int(center[2][0])-10, int(center[2][1])-10), (int(center[2][0])+10, int(center[2][1])+10), (0, 255, 0), 5)
        cv2.line(vis, (int(center[2][0])+10, int(center[2][1])-10), (int(center[2][0])-10, int(center[2][1])+10), (0, 255, 0), 5) 

        # scale back the corners to real size of orthomosaic geotiff
        count=0
        for pt in corners:
            # pt[0] = int(pt[0]/scale_x)
            # pt[1] = int(pt[1]/scale_y)ssssssssssssssssssssssssssssssss
            cv.circle(vis ,(int(pt[0]), int(pt[1])), 13, (255, 0, 0), 3)

            # if not logger is None:
            #     logger.info('Reprojected %s corner px is: %s, %s', text, str(pt[0]), str(pt[1]))

            #cv2.putText(vis, text, (int(pt[0])+20, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
            count=count+1

        #corners = np.int32(corners)
        #cv.polylines(vis, [corners], True, (255, 0, 0), 3)

    f = None
    if IsDumpPtPair:
        if not kp_pairs is None:
            f = open(os.path.join(output_dir, "all_asift_pair.txt"), "a")
            f.write("ortho_x ortho_y persp_x persp_y\n")
            if not logger is None:
                logger.info('Start to dump all asift matching pair pixel coordinate to a file...')
        else:
            f = open(os.path.join(output_dir, "final_pair.txt"), "a")
            f.write("ortho_x ortho_y ortho_lat ortho_lon persp_x persp_y\n")
            if not logger is None:
                logger.info('Start to dump all final matching pair pixel coordinate to a file...')
        
        

    p1, p2 = [], [] 
    if not kp_pairs is None:
        status = np.ones(len(kp_pairs), np.bool_)
        for kpp in kp_pairs:
            p1.append(np.int32(kpp[0].pt))
            p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

            if IsDumpPtPair:
                f.write(str(int(kpp[0].pt[0])) + ' ' + str(int(kpp[0].pt[1])) + ' ' + str(int(kpp[1].pt[0])) + ' ' + str(int(kpp[1].pt[1])) + "\n")
    
    else:
        status = np.ones(len(pp1), np.bool_)
        for ii in range(0, len(pp1)):
            p1.append(np.int32(pp1[ii]))
            p2.append(np.int32(np.array(pp2[ii]) + [w1, 0]))  

            if IsDumpPtPair and not pp1_gps is None:
                f.write(str(int(pp1[ii][0])) + ' ' + str(int(pp1[ii][1])) + ' ' + str(pp1_gps[ii][0]) + ' ' + str(pp1_gps[ii][1]) + ' ' + str(int(pp2[ii][0])) + ' ' + str(int(pp2[ii][1])) + "\n")          

    if IsDumpPtPair:
        f.close()      

    green = (0, 255, 0)
    red = (0, 0, 255)
    kp_color = (51, 103, 236)
    circie_radius = 3

    for count, ((x1, y1), (x2, y2), inlier) in enumerate(zip(p1, p2, status)):

        if inlier:
            if count+Num_of_additional_point_pair >= len(status):
                col = red
            else:
                col = green
            cv.circle(vis, (x1, y1), circie_radius, col, -1)
            cv.circle(vis, (x2, y2), circie_radius, col, -1)

        elif not isOnlyKeyPt:
            col = red
            r = 2
            thickness = 3
            cv.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    
    counter=0
    ShowRegionSize=100
    cross_length=8
    if not isOnlyKeyPt:
        for count, ((x1, y1), (x2, y2), inlier) in enumerate(zip(p1, p2, status)):
            if inlier:
                if count+Num_of_additional_point_pair >= len(status):
                    col = red
                else:
                    col = green
                cv.line(vis, (x1, y1), (x2, y2), col)

                # Also saved the matched inliers small photo for debug purpose
                if not output_dir is None and kp_pairs is None:
                    if x1 > ShowRegionSize*0.5 and x1 < vis.shape[1] - ShowRegionSize*0.5 and \
                        y1 > ShowRegionSize*0.5 and y1 < vis.shape[0] - ShowRegionSize*0.5 and \
                        x2 > ShowRegionSize*0.5 and x2 < vis.shape[1] - ShowRegionSize*0.5 and \
                        y2 > ShowRegionSize*0.5 and y2 < vis.shape[0] - ShowRegionSize*0.5:

                        matched_features = np.zeros((ShowRegionSize, ShowRegionSize*2), np.uint8)
                        matched_features = cv.cvtColor(matched_features, cv.COLOR_GRAY2BGR)
                        matched_features[:ShowRegionSize, :ShowRegionSize] = vis_copy[y1-50:y1+50, x1-50:x1+50]
                        matched_features[:ShowRegionSize, ShowRegionSize:ShowRegionSize*2] = vis_copy[y2-50:y2+50, x2-50:x2+50]

                        # draw a red cross to represent the centre
                        cv2.line(matched_features, (int(ShowRegionSize*0.5)-cross_length, int(ShowRegionSize*0.5)-cross_length), (int(ShowRegionSize*0.5)+cross_length, int(ShowRegionSize*0.5)+cross_length), (0, 0, 255), 1)
                        cv2.line(matched_features, (int(ShowRegionSize*0.5)+cross_length, int(ShowRegionSize*0.5)-cross_length), (int(ShowRegionSize*0.5)-cross_length, int(ShowRegionSize*0.5)+cross_length), (0, 0, 255), 1)

                        cv2.line(matched_features, (int(ShowRegionSize+ShowRegionSize*0.5)-cross_length, int(ShowRegionSize*0.5)-cross_length), (int(ShowRegionSize+ShowRegionSize*0.5)+cross_length, int(ShowRegionSize*0.5)+cross_length), (0, 0, 255), 1)
                        cv2.line(matched_features, (int(ShowRegionSize+ShowRegionSize*0.5)+cross_length, int(ShowRegionSize*0.5)-cross_length), (int(ShowRegionSize+ShowRegionSize*0.5)-cross_length, int(ShowRegionSize*0.5)+cross_length), (0, 0, 255), 1)

                        cv.imwrite(os.path.join(output_dir, "asift_inlier_pair_" + str(counter) + ".png"), matched_features)
                        counter=counter+1


    return vis

def filter_matches(kp1, kp2, matches, ratio = 0.75):

    """
    function to select feature detector and matcher

    Note:
        N/A

    Args:
        kp1 (opencv define type): key points of orthophoto
        kp2 (opencv define type): key points of perspective image
        matches (opencv define type): matching pairs of key points
        ratio (double): SIFT ratio to filter out false matchings

    Returns:
        detector (N/A): feature detector
        matcher (N/A): feature matcher

    Raises:
        N/A
    
    """

    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]            
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

def marker_matching_by_hue_vector(result_contour_ROI, result_contour_id, lowerhsvlist, upperhsvlist, frame_hsv_original_copy, frame_bwh_original_copy, 
        frame_bwl_original_copy, result_contour_img, logger):

    """
    function to match the marker candidate on 2 sides by using hue vector

    Note:
        N/A

    Args:
        result_contour_ROI (nparray): The marker contour candidate ROI
        result_contour_id (nparray): The marker contour candidate id
        lowerhsvlist (nparray): The lower threshold of the HSV value for major colors
        upperhsvlist (nparray): The upper threshold of the HSV value for major colors
        frame_hsv_original_copy (nparray): The original size hsv image of both perspective and orthophoto
        frame_bwh_original_copy (nparray): The original size binary image after high white threshold of both perspective and orthophoto
        frame_bwl_original_copy (nparray): The original size binary image after low white threshold of both perspective and orthophoto
        result_contour_img (nparray): The marker contour candidate image(cropped)
        logger (python object): The logger handler for logging

    Returns:
        matched_indices (nparray): The matched indices of the marker id pair

    Raises:
        N/A
    
    """

    huevector = [None] * 2
    huevector[0] = [None] * len(result_contour_id[0])
    huevector[1] = [None] * len(result_contour_id[1])

    # Count the non zero pixel in each color, and contruct a vector having all the sum of different color
    for qq in range(0, len(huevector)):
        for bb in range(0, len(result_contour_ROI[qq])):
            area_of_ROI = result_contour_ROI[qq][bb][2]*result_contour_ROI[qq][bb][3]
            logger.info('area_of_ROI: %s', str(area_of_ROI))
            
            area_in_hue = [0.0] * (len(lowerhsvlist)+2)
            x,y,w,h = result_contour_ROI[qq][bb]
            logger.info('ROI: %s, %s, %s, %s', str(x), str(y), str(w), str(h))

            #color
            for yy in range(0, len(lowerhsvlist)):
                hsvmask = cv2.inRange(frame_hsv_original_copy[qq][y:y+h,x:x+w], lowerhsvlist[yy], upperhsvlist[yy])
                #hsvmask = cv2.morphologyEx(hsvmask, cv2.MORPH_OPEN, morph_open_kernel)
                area_of_non_zero = cv2.countNonZero(hsvmask)
                logger.info('qq: %s  idx: %s  area_of_non_zero: %s', str(qq), str(result_contour_id[qq][bb]), str(area_of_non_zero))
                #area_in_hue[yy] = 1.0*area_of_non_zero/area_of_ROI
                area_in_hue[yy] = 1.0*area_of_non_zero

            #white and black
            area_of_non_zero = cv2.countNonZero(frame_bwh_original_copy[qq][y:y+h,x:x+w])
            area_in_hue[len(lowerhsvlist)] = area_of_non_zero
            logger.info('qq: %s  idx: %s  area_of_non_zero: %s', str(qq), str(result_contour_id[qq][bb]), str(area_of_non_zero))

            area_of_non_zero = cv2.countNonZero(frame_bwl_original_copy[qq][y:y+h,x:x+w])
            area_in_hue[len(lowerhsvlist)+1] = area_of_non_zero
            logger.info('qq: %s  idx: %s  area_of_non_zero: %s', str(qq), str(result_contour_id[qq][bb]), str(area_of_non_zero))

            dst = np.asarray([0.0] * len(area_in_hue))
            cv2.normalize(np.asarray(area_in_hue), dst, 1.0, 0.0, cv2.NORM_L2)
            area_in_hue = dst
            
            huevector[qq][bb] = np.asarray(area_in_hue)
            logger.info('area_in_hue: %s', str(area_in_hue))


    logger.info('huevector: %s', str(huevector))

    # Do the matching using Hungarian algorithm
    dist_matrix = np.zeros((len(result_contour_img[0]),len(result_contour_img[1])),dtype=np.float32)
    for d,img1 in enumerate(result_contour_img[0]):
        for t,img2 in enumerate(result_contour_img[1]):
            # with itself, score=0
            dist_matrix[d,t] = cv2.norm(huevector[0][d], huevector[1][t])

    logger.info('dist_matrix: %s', str(dist_matrix))
    matched_indices = linear_assignment(dist_matrix)
    logger.info('matched_indices: %s', str(matched_indices))

    for matches in matched_indices:
        logger.info('matching pair( %s, %s)', str(result_contour_id[0][matches[0]]), str(result_contour_id[1][matches[1]]))
    
    return matched_indices

def output_matching_img(frame_img, output_dir, final_src_pts, final_dst_pts):

    """
    function to save the final matching img for debugging purpose

    Note:
        N/A

    Args:
        frame_img (nparray): The original color image of perspective and ortho
        output_dir (string): The path of the output directory of the matching image
        final_src_pts (nparray): The correct ordered marker corner point in the perspective image
        final_dst_pts (nparray): The correct ordered marker corner point in the ortho image

    Returns:
        N/A

    Raises:
        N/A
    
    """

    # Print out resized rgb image with the point index to see if the matching is correct
    displayimg2 = frame_img[0].copy()
    for jjj in range(0, len(final_src_pts)):
        cv2.putText(displayimg2,str(jjj), (int(final_src_pts[jjj][0][0]) ,int(final_src_pts[jjj][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.circle(displayimg2,(int(final_src_pts[jjj][0][0]) ,int(final_src_pts[jjj][0][1])), 5, (0, 255, 0)) 


    displayimg3 = frame_img[1].copy()
    for jjj in range(0, len(final_dst_pts)):
        cv2.putText(displayimg3,str(jjj), (int(final_dst_pts[jjj][0][0]), int(final_dst_pts[jjj][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.circle(displayimg3,(int(final_dst_pts[jjj][0][0]) ,int(final_dst_pts[jjj][0][1])), 5, (0, 255, 0))
    #cv2.imwrite(os.path.join(output_dir, 'i=' + str(1) + '_raw_color_matched_index.jpg'), displayimg2)


    h1, w1 = displayimg2.shape[:2]
    h2, w2 = displayimg3.shape[:2]

    #create empty matrix
    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

    #combine 2 images
    vis[:h1, :w1,:3] = displayimg2
    vis[:h2, w1:w1+w2,:3] = displayimg3

    cv2.imwrite(os.path.join(output_dir, 'zzz_raw_color_matched_index.jpg'), vis)

def determine_corner_point_order(matched_indices, result_contour_cnt, result_contour_id, logger):

    """
    function to determine the correct order of matching of the corner points after the matching of marker

    Note:
        N/A

    Args:
        matched_indices (nparray): The matched indices of the marker on both side of perspective and orthophoto
        result_contour_cnt (nparray): The marker contour point list
        result_contour_id (nparray): The marker contour id
        logger (python object): The logger handler for logging

    Returns:
        src_pts (nparray): test pt(i.e. centroid of marker) in the perspective image
        dst_pts (nparray): test pt(i.e. centroid of marker) in the orthophoto image
        final_src_pts (nparray): final marker corner points in a correct order to match with final_dst_pts
        final_dst_pts (nparray): final marker corner points in a correct order to match with final_src_pts

    Raises:
        N/A
    
    """
    delta_idx=[0]*len(matched_indices) # only 4 combinations for 4 corners points


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

    # main loop, start to determine the corner points order for each matched marker pair
    for matches in matched_indices:
        bestidx=0
        besterror=1e9
        bestcopylist=None
        copylist= result_contour_cnt[1][matches[1]]

        logger.info('Guessing delta_idx for matches: %s %s', str(result_contour_id[0][matches[0]]), str(result_contour_id[1][matches[1]]))

        # calculate the reprojection error of the center of mass, which has a unique answer
        # The order only has 4 possiblilites, so try each out in a round table manner
        for roundj in range(0, 4):

            if roundj is not 0:
                endvalue = copylist[-1]
                copylist = np.insert(copylist[:-1], 0, endvalue, axis=0)

            logger.info('copylist: %s', str(copylist.flatten()))

            #homography, mask = cv2.findHomography(result_contour_cnt[0][matches[0]], copylist, cv2.RANSAC,ransacReprojThreshold)
            mixed_src = np.concatenate((result_contour_cnt[0][matches[0]], src_pts))
            mixed_dst = np.concatenate((copylist, dst_pts))

            #In current order, calculate a homography out of it
            homography, mask = cv2.findHomography(mixed_src, mixed_dst, 0)
            #print("mask: ", mask.transpose())

    
            reprojectimgpts = cv2.perspectiveTransform(src_pts, homography)
            reprojectionerror=0
            errorlist=[]

            for zz in range(0, len(src_pts)):
                error = math.sqrt(math.pow(reprojectimgpts[zz][0][0]-dst_pts[zz][0][0], 2)+math.pow(reprojectimgpts[zz][0][1]-dst_pts[zz][0][1], 2))
                errorlist.append(error)
                reprojectionerror = reprojectionerror + error

            logger.info('errorlist= %s', str(errorlist))
            logger.info('reprojectionerror: %s', str(reprojectionerror))

            # Let the order with least error to be the output order
            if reprojectionerror < besterror:
                besterror = reprojectionerror
                bestidx = roundj
                bestcopylist = copylist.copy()


        delta_idx[counttt] = bestidx
        logger.info('bestidx: %s', str(bestidx))
        logger.info('besterror: %s', str(besterror))
        logger.info('=======================================================================')
        counttt=counttt+1

        if final_src_pts is None:
            final_src_pts = np.asarray(result_contour_cnt[0][matches[0]])
        else:
            final_src_pts = np.append(final_src_pts, result_contour_cnt[0][matches[0]], axis=0)


        if final_dst_pts is None:
            final_dst_pts = np.asarray(bestcopylist.copy())
        else:
            final_dst_pts = np.append(final_dst_pts, bestcopylist, axis=0)

    logger.info('delta_idx: %s', str(delta_idx))
    #print("final_dst_pts: ", final_dst_pts)

    #print("final_src_pts: ", final_src_pts*scaledownratio[0])
    #print("final_dst_pts: ", final_dst_pts*scaledownratio[1])

    return final_src_pts, final_dst_pts

def corner_refinement(i, obj_id, x, y, m, approx, imgg, imgg_color, mergedlineminthreshold, 
       scalethd_for_intersected_lsd, RefinedCornerThreshold, sameline_thd, 
       lines_merging_min_length, lines_merging_angle_diff_in_degree, lines_merging_extension_ratio, lines_merging_maxgap, 
       isMarkerDebug, logger, output_dir):

    """
    function to refine the corner of the marker initial contour corner point

    Note:
        N/A

    Args:
        i (int): 0: perspective, 1: ortho
        obj_id (int): index of the marker candidate, for debug purpose
        x (int): Upper-left corner x coordinate of the bounding box
        y (int): Upper-left corner y coordinate of the bounding box
        m (int): margin of the bounding box
        approx (nparray): original contour of the marker
        imgg (nparray): The chopped rectangle region of the marker contour in gray color
        imgg_color (nparray): The chopped rectangle region of the marker contour in color
        mergedlineminthreshold (double): threshold to filter line by length after the merged line
        scalethd_for_intersected_lsd (double): All merged lines will compute all possible intersections, it is a threshold similar to meaning of lines_merging_extension_ratio
        RefinedCornerThreshold (double): original corner will search for refined corner in this distance range
        sameline_thd (double): Judge whether 2 lines are the same by the distance of their endpoints in px, duplicate line will be remove
        lines_merging_min_length (double): entrance threshold for the min length line that accept merging
        lines_merging_angle_diff_in_degree (double): threshold angle between the 2 lines to accept the merging
        lines_merging_extension_ratio (double): threshold for how much the 2 lines endpoint need to extend (in ratio) to their intersection point
        lines_merging_maxgap (double): threshold for how much the 2 lines endpoint need to extend (in pixel) to their intersection point
        logger (python object): min y value in gps coordinate
        output_dir (string): The path to output intermediate photos

    Returns:
        approx_new (nparray): new contour of the marker in which some corners are refined.

    Raises:
        N/A
    
    """
    
    img_lsd = imgg_color.copy()
    img_lsd_filter = imgg_color.copy()
    #Create default parametrization LSD
    #LSD: Line Segment Detector is a module in opencv
    lsd = cv2.createLineSegmentDetector(0)
    #lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV, 1.0, )

    #Detect lines in the image
    lineslsd = lsd.detect(imgg)[0] #Position 0 of the returned tuple are the detected lines
    img_lsd = lsd.drawSegments(img_lsd,lineslsd)
    logger.info('len(lineslsd): %s', str(len(lineslsd)))
    #print("lineslsd: ", lineslsd)

    if isMarkerDebug:  
        cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_lsd.jpg'), img_lsd)   

    img_lsd_merge = imgg_color.copy()

    # Merge short segments into possible longer segment
    line_merged = lines_merging(lineslsd, lines_merging_min_length, lines_merging_angle_diff_in_degree, lines_merging_extension_ratio, lines_merging_maxgap)

    if len(line_merged)>0:
        #img_lsd_merge = lsd.drawSegments(img_lsd_merge,np.asarray(line_merged))
        for merge_line in line_merged:
            cv2.line(img_lsd_merge,(int(merge_line[0][0]), int(merge_line[0][1])),(int(merge_line[0][2]), int(merge_line[0][3])),(0,0,255),1)  

        if isMarkerDebug:  
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_lsd_merge.jpg'), img_lsd_merge)   

    #Non maximal suppression on merged segment to remove duplicate
    line_merged_nms=[]
    for merge_line in line_merged:
        isRepeat=False
        for merge_line2 in line_merged:
            if not merge_line is merge_line2 and checksameline(merge_line[0], merge_line2[0], sameline_thd):
                isRepeat = True

        if not isRepeat:
            line_merged_nms.append(merge_line)
        else:
            isExist=False
            for linee in line_merged_nms:
                if checksameline(linee[0], merge_line[0], sameline_thd):
                    isExist=True
            
            if not isExist:
                line_merged_nms.append(merge_line)

    logger.info('len(line_merged_nms): %s', str(len(line_merged_nms)))
    #print("line_merged_nms: ", np.asarray(line_merged_nms))

    img_lsd_merge_thd_list=[]
    img_lsd_merge_thd = imgg_color.copy()
    for merge_line in line_merged_nms:
        length = linelength(merge_line[0])
        if length > mergedlineminthreshold:  
            img_lsd_merge_thd_list.append(merge_line)
            cv2.line(img_lsd_merge_thd,(int(merge_line[0][0]), int(merge_line[0][1])),(int(merge_line[0][2]), int(merge_line[0][3])),(0,0,255),1)  

        if isMarkerDebug:
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_lsd_merge_thd.jpg'), img_lsd_merge_thd)                                                        
    logger.info('len(img_lsd_merge_thd_list): %s', str(len(img_lsd_merge_thd_list)))                    

    img_lsd_merge_intersect = imgg_color.copy()
    convexhullpointlist=[]
    idx1=0

    # Calculate all the intersection points of all the long enough merged lines
    if len(line_merged_nms)>0:
        #img_lsd_merge = lsd.drawSegments(img_lsd_merge,np.asarray(line_merged))
        for merge_line in line_merged_nms:
            idx2=0
            length = linelength(merge_line[0])
            if length > mergedlineminthreshold:  
                for merge_line2 in line_merged_nms:
                    length2 = linelength(merge_line2[0])        
                    if length2 > mergedlineminthreshold:  
                        if idx2 > idx1 and not checksameline(merge_line[0], merge_line2[0], sameline_thd):  
                            #print("Line for intersect: ", merge_line[0], merge_line2[0])
                            (xi, yi, valid, r, s) = intersect(merge_line[0], merge_line2[0])   
                            if valid == 1 and min(math.fabs(r-1.0), math.fabs(r)) < scalethd_for_intersected_lsd and min(math.fabs(s-1.0), math.fabs(s)) < scalethd_for_intersected_lsd:
                                convexhullpointlist.append([[xi, yi]])
                                cv2.circle(img_lsd_merge_intersect, (int(xi), int(yi)), 5, (0,255,0), 2)

                    idx2=idx2+1

            idx1=idx1+1

        if isMarkerDebug:
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_lsd_mintersect.jpg'), img_lsd_merge_intersect)                       


    img_lsd_convexhull = imgg_color.copy()

    # Do a convex hull to preserve only the outside intersection points
    if len(convexhullpointlist)>0:
        hull = cv2.convexHull(np.asarray(convexhullpointlist, dtype=np.float32))
        logger.info('len(hull): %s', str(len(hull)))   
        for pt in hull:
            #print("pt: ", pt)
            cv2.circle(img_lsd_convexhull, (int(pt[0][0]), int(pt[0][1])), 5, (0,255,0), 2)
        
        if isMarkerDebug:
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_idx_' + str(obj_id) + '_lsd_mzconvexhull.jpg'), img_lsd_convexhull)  
    
    
    # Finally start to refine the corner points
    if len(hull) > 0:
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

            # Refine it if it is within certain distance from the original corner point
            if min_dist < RefinedCornerThreshold:
                logger.info('original pt %s is refined by %s with dist= %s', str(pt), str(hull_in_globalcoord[min_dist_index]), str(min_dist))
                approx_new.append(hull_in_globalcoord[min_dist_index])
            else:
                logger.info('No replacement, min dist= %s', str(min_dist))
                approx_new.append(pt)
        
        logger.info('old approx: %s', str(approx.flatten()))
        logger.info('new approx: %s', str(np.asarray(approx_new).flatten()))
                                
    return approx_new

def detect_markers(marker_config_file, logger, img1_path, img2_path, isMarkerDebug, orthomosaicscale):

    """
    function to detect artificial markers placed on the ground and give extra points pair for the homography fitting

    Note:
        N/A

    Args:
        marker_config_file (string): path to the config file containing the parameters need to do marker detection and matching
        logger: logger handler from the main function for logging
        img1_path (string):orthophoto path from main function with original size
        img2_path (string):persepctive photo path from main function with original size
        isMarkerDebug (bool): A flag to control whether marker detection run in debug mode
        orthomosaicscale (double): m/px scale for orthomosaic image

    Returns:
        src_pt, dst_pt (nparray, nparray): src and dst point pair in px

    Raises:
        N/A
    
    """

    fs_read = None
    try:
        fs_read = cv2.FileStorage(marker_config_file, cv2.FILE_STORAGE_READ)
    except:
        logger.error("fail to load marker config file")
        exit(1)
    
   

    MarkerSize = fs_read.getNode("MarkerSize_in_m").real()
    marker_area_variation = fs_read.getNode("marker_area_variation").real()
    area_to_perimeter_ratio_threshold = fs_read.getNode("area_to_perimeter_ratio_threshold").real()
    sameline_thd = fs_read.getNode("sameline_thd_in_px").real()
    lines_merging_min_length = fs_read.getNode("lines_merging_min_length_in_px").real()
    lines_merging_angle_diff_in_degree = fs_read.getNode("lines_merging_angle_diff_in_degree").real()
    lines_merging_extension_ratio = fs_read.getNode("lines_merging_extension_ratio").real()
    lines_merging_maxgap = fs_read.getNode("lines_merging_maxgap_in_px").real()
    aspect_ratio_thd = fs_read.getNode("aspect_ratio_thd").real()
    approxPolyDP_arclength_ratio = fs_read.getNode("approxPolyDP_arclength_ratio").real()
    scalethd_for_intersected_lsd = fs_read.getNode("scalethd_for_intersected_lsd").real()


    # Parameter that are different beteween perspective img and orthophoto img
    img_path_list=[img2_path, img1_path]
    thrhd_grayvalue_high_for_whitepart_of_chessboard_per = int(fs_read.getNode("thrhd_grayvalue_high_for_whitepart_per").real())
    thrhd_grayvalue_high_for_whitepart_of_chessboard_ortho = int(fs_read.getNode("thrhd_grayvalue_high_for_whitepart_ortho").real())
    thrhd_grayvalue_high_for_whitepart_of_chessboard=[thrhd_grayvalue_high_for_whitepart_of_chessboard_per,thrhd_grayvalue_high_for_whitepart_of_chessboard_ortho] 

    thrhd_grayvalue_low_for_whitepart_of_chessboard_per = int(fs_read.getNode("thrhd_grayvalue_low_for_darkpart_per").real())
    thrhd_grayvalue_low_for_whitepart_of_chessboard_ortho = int(fs_read.getNode("thrhd_grayvalue_low_for_darkpart_ortho").real())
    thrhd_grayvalue_low_for_whitepart_of_chessboard=[thrhd_grayvalue_low_for_whitepart_of_chessboard_per,thrhd_grayvalue_low_for_whitepart_of_chessboard_ortho]

    RefinedCornerThreshold_per = int(fs_read.getNode("RefinedCornerThreshold_per").real())
    RefinedCornerThreshold_ortho = int(fs_read.getNode("RefinedCornerThreshold_ortho").real())
    RefinedCornerThreshold=[RefinedCornerThreshold_per,RefinedCornerThreshold_ortho]

    boundingrectmargin_per = int(fs_read.getNode("boundingrectmargin_in_px_per").real())
    boundingrectmargin_ortho = int(fs_read.getNode("boundingrectmargin_in_px_ortho").real())
    boundingrectmargin=[boundingrectmargin_per,boundingrectmargin_ortho]

    mergedlineminthreshold_per = int(fs_read.getNode("mergedlineminthreshold_in_px_per").real())
    mergedlineminthreshold_ortho = int(fs_read.getNode("mergedlineminthreshold_in_px_ortho").real())
    mergedlineminthreshold=[mergedlineminthreshold_per,mergedlineminthreshold_ortho]

    # dont scale down survelliance camera image, since marker at the far end will be very small
    scaledownratio_orthophoto = fs_read.getNode("scaledownratio_orthophoto").real()
    scaledownratio_perspective = fs_read.getNode("scaledownratio_perspective").real()
    scaledownratio = [scaledownratio_orthophoto,scaledownratio_perspective]

    IsFindDarkAlsoNearSegmentedWhite=eval(fs_read.getNode("IsFindDarkAlsoNearSegmentedWhite").string())
    IsFindWhiteOnlyNearColorSeg=eval(fs_read.getNode("IsFindWhiteOnlyNearColorSeg").string())
    IsUsingRefinedCorner=eval(fs_read.getNode("IsUsingRefinedCorner").string())
    IsHSVspecificTuning=eval(fs_read.getNode("IsHSVspecificTuning").string())



    length_approx_threshold_low = int(fs_read.getNode("length_approx_threshold_low").real())
    length_approx_threshold_high = int(fs_read.getNode("length_approx_threshold_high").real())
    morph_open_kernel_radius_in_px = int(fs_read.getNode("morph_open_kernel_radius_in_px").real())
    morph_close_kernel_radius_in_px = int(fs_read.getNode("morph_close_kernel_radius_in_px").real())
    dilate_kernel_size_in_px_for_cb_mask_radius = int(fs_read.getNode("dilate_kernel_size_in_px_for_cb_mask_radius").real())
    dilate_kernel_size_for_white = int(fs_read.getNode("dilate_kernel_size_for_white").real())
    approxPolyDP_epsilon = int(fs_read.getNode("approxPolyDP_epsilon").real())
    area_lower_threshold = int(fs_read.getNode("area_lower_threshold_in_px").real())
    area_upper_threshold = int(fs_read.getNode("area_upper_threshold_in_px").real())
    reflective_surface_thd = int(fs_read.getNode("reflective_surface_thd").real())

    #Define common color HSV lower and upper
    S_lower = int(fs_read.getNode("S_lower").real())
    S_Upper = int(fs_read.getNode("S_Upper").real())
    V_lower = int(fs_read.getNode("V_lower").real())
    V_Upper = int(fs_read.getNode("V_Upper").real())
    H_lower = int(fs_read.getNode("H_lower").real())
    H_Upper = int(fs_read.getNode("H_Upper").real())

    #From 0 to 180deg
    redhsv = int(fs_read.getNode("redhsv").real())
    red2hsv = int(fs_read.getNode("red2hsv").real())
    yellowhsv = int(fs_read.getNode("yellowhsv").real())
    greenhsv = int(fs_read.getNode("greenhsv").real())
    tealhsv = int(fs_read.getNode("tealhsv").real())
    bluehsv = int(fs_read.getNode("bluehsv").real())
    purplehsv = int(fs_read.getNode("purplehsv").real())

    ###### Input parameters ends ######

    colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

    morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_open_kernel_radius_in_px,morph_open_kernel_radius_in_px))   
    morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_close_kernel_radius_in_px,morph_close_kernel_radius_in_px))   
    dilate_kernel_for_cb_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate_kernel_size_in_px_for_cb_mask_radius, dilate_kernel_size_in_px_for_cb_mask_radius))    
    dilate_kernel_white = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate_kernel_size_for_white, dilate_kernel_size_for_white))    


    colorlist=[red2hsv, yellowhsv, greenhsv, tealhsv, bluehsv, purplehsv]

    lowerhsvlist=[]
    upperhsvlist=[]

    # Setup the upper and lower HSV threshold range for major colors
    for color in colorlist:
        lower = np.array([max(color+H_lower, 0),S_lower,V_lower])
        lowerhsvlist.append(lower)
        upper = np.array([min(color+H_Upper, 180),S_Upper,V_Upper])
        upperhsvlist.append(upper)

    if IsHSVspecificTuning:
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


    logger.info('==================== Marker detection and matching start ==========================')
    logger.info('img_path_list: %s', str(img_path_list))
    logger.info('MarkerSize(in m): %s', str(MarkerSize))
    logger.info('orthomosaicscale(m/px): %s', str(orthomosaicscale))
    logger.info('marker_area_variation: %s', str(marker_area_variation))
    logger.info('area_to_perimeter_ratio_threshold: %s', str(area_to_perimeter_ratio_threshold))
    logger.info('IsUsingRefinedCorner: %s', str(IsUsingRefinedCorner))
    logger.info('IsFindDarkAlsoNearSegmentedWhite: %s', str(IsFindDarkAlsoNearSegmentedWhite))
    logger.info('IsFindWhiteOnlyNearColorSeg: %s', str(IsFindWhiteOnlyNearColorSeg))
    logger.info('length_approx_threshold_low: %s', str(length_approx_threshold_low))
    logger.info('length_approx_threshold_high: %s', str(length_approx_threshold_high))
    logger.info('boundingrectmargin: %s', str(boundingrectmargin))
    logger.info('reflective_surface_thd: %s', str(reflective_surface_thd))
    logger.info('RefinedCornerThreshold: %s', str(RefinedCornerThreshold))
    logger.info('length_approx_threshold_high: %s', str(length_approx_threshold_high))
    logger.info('thrhd_grayvalue_high_for_whitepart_of_chessboard: %s', str(thrhd_grayvalue_high_for_whitepart_of_chessboard))
    logger.info('thrhd_grayvalue_low_for_whitepart_of_chessboard: %s', str(thrhd_grayvalue_low_for_whitepart_of_chessboard))
    logger.info('morph_open_kernel_radius_in_px: %s', str(morph_open_kernel_radius_in_px))
    logger.info('morph_close_kernel_radius_in_px: %s', str(morph_close_kernel_radius_in_px))
    logger.info('dilate_kernel_size_in_px_for_cb_mask_radius: %s', str(dilate_kernel_size_in_px_for_cb_mask_radius))
    logger.info('dilate_kernel_size_for_white: %s', str(dilate_kernel_size_for_white))
    logger.info('approxPolyDP_epsilon: %s', str(approxPolyDP_epsilon))
    logger.info('scaledownratio: %s', str(scaledownratio))
    logger.info('lowerhsvlist: %s', str(lowerhsvlist))
    logger.info('upperhsvlist: %s', str(upperhsvlist))


    output_dir=None

    # Open a seperate folder with date to put in the debugging file
    if isMarkerDebug:
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

        output_dir = "./MarkerDetection_Debug_"+str(st) +"/"
        os.mkdir(output_dir)

    #calculate the expected physical size of marker for orthomosaic image

    side_in_px = (MarkerSize)/(orthomosaicscale*scaledownratio[1])
    logger.info('[GeoTiff] expected marker side in px (in resized img):  %f', side_in_px)
    area_in_px = side_in_px*side_in_px
    logger.info('[GeoTiff] expected marker area in px (in resized img):  %f', area_in_px)

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

    # Main loop for survelliance (0) and orthomosaic(1)
    for i in range(0, len(img_path_list)):
        img_path = img_path_list[i]

        logger.info('///////////////////////////////////////////////////')
        if i is 0:
            logger.info('Doing survelliance camera round...')
        else:
            logger.info('Doing orthomosaic image round...')

        # Reading image 
        img2 = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        logger.info('input image original size:  %s', str(img2.shape))

        # converting to gray scale. 

        logger.info('color to gray image...')
        frame_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)         

        if isMarkerDebug:
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_1raw_color.jpg'), img2)

        img2 = cv2.resize(img2, (0,0), fx=(1/scaledownratio[i]), fy=(1/scaledownratio[i])) 
        logger.info('input image resized size:  %s', str(img2.shape))

        frame_img2[i] = img2.copy()

        if isMarkerDebug:
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_2raw_color_resized.jpg'), img2)

        # convert to HSV image

        imghsv = img2.copy()
        imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2HSV ) 
        logger.info('Convert image to HSV...')
        frame_hsv_original_copy[i] = imghsv.copy()

        if isMarkerDebug:
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_3raw_hsv_resized.jpg'), imghsv)
        
        # Reading same image in another variable and  


        logger.info('Resize gray image') 
        frame_gray = cv2.resize(frame_gray, (0,0), fx=(1/scaledownratio[i]), fy=(1/scaledownratio[i])) 

        if isMarkerDebug:
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_4raw_gray_resized.jpg'), frame_gray)    


        logger.info('Set highly reflective surface as 0...')

        # since marker white is not reflective surface, will not be very close to 255
        frame_gray[frame_gray > reflective_surface_thd] = 0

        hsvmask=None
        logger.info('Start color segmentation...')
        for j in range(0, len(lowerhsvlist)):
            newhsvmask = cv2.inRange(imghsv, lowerhsvlist[j], upperhsvlist[j])

            if isMarkerDebug:     
                cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_5after' + str(j) + 'hsv_in_' + str(namestr(colorlist[j], globals())) + '.jpg'), newhsvmask)

            if hsvmask is None:
                hsvmask = newhsvmask.copy()
            else:
                # bitwise_or to combine all segmentation result from different colors
                hsvmask = cv2.bitwise_or(newhsvmask, hsvmask)   

        hsvmask = cv2.morphologyEx(hsvmask, cv2.MORPH_CLOSE, morph_open_kernel)

        if isMarkerDebug:  
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_6after_color_seg.jpg'), hsvmask)



        logger.info('End color segmentation...')

        # Color segmentation for black and white color solely


        bw_ret,frame_bwh_original = cv2.threshold(frame_gray,thrhd_grayvalue_high_for_whitepart_of_chessboard[i],255,cv2.THRESH_BINARY)

        if isMarkerDebug:  
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_7threshold_white_original.jpg'), frame_bwh_original)    

        frame_bw=frame_bwh_original.copy()
        frame_bwh_original_copy[i] = frame_bwh_original.copy()

        bw_ret,frame_bwl_original = cv2.threshold(frame_gray,thrhd_grayvalue_low_for_whitepart_of_chessboard[i],255,cv2.THRESH_BINARY_INV)
        frame_bwl_original_copy[i] = frame_bwl_original.copy()    

        if isMarkerDebug:  
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_8threshold_dark_original.jpg'), frame_bwl_original)    

        # Old mode, since dark is very sensitive and make many false alarm, to find dark in the nearby area of segmented white
        if IsFindDarkAlsoNearSegmentedWhite:
            
            frame_bwh = cv2.morphologyEx(frame_bwh_original, cv2.MORPH_OPEN, morph_open_kernel)
            frame_bwh = cv2.dilate(frame_bwh,dilate_kernel_for_cb_mask,iterations = 1)

            frame_bwh_dilate = cv2.dilate(frame_bwh,dilate_kernel_white,iterations = 1)

            if isMarkerDebug:  
                cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_9dilate_w_as_mask.jpg'), frame_bwh_dilate)


            # set 255 to 0 since orthomosaic has white boundary
            for y in range(len(frame_gray)):
                for x in range(len(frame_gray[0])):
                    if frame_gray[y][x] == 0:
                        frame_gray[y][x] = 255


            frame_bwl = cv2.morphologyEx(frame_bwl_original, cv2.MORPH_CLOSE, morph_close_kernel)
            frame_bwl = cv2.dilate(frame_bwl,dilate_kernel_for_cb_mask,iterations = 1)



            frame_bw = cv2.bitwise_or(frame_bwh_original, frame_bwl, mask=frame_bwh_dilate)   

            if isMarkerDebug:  
                cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_10white_OR_dark.jpg'), frame_bw)
            logger.info('Merging white and dark segmentation...')


        
        color_or_bw=None

        # Old mode, similar rationale for the above IsFindDarkAlsoNearSegmentedWhite
        if IsFindWhiteOnlyNearColorSeg:
            hsvmask_asmask = cv2.morphologyEx(hsvmask, cv2.MORPH_OPEN, morph_open_kernel)
            hsvmask_asmask = cv2.dilate(hsvmask_asmask,dilate_kernel_for_cb_mask,iterations = 1)        
            hsvmask_asmask = cv2.dilate(hsvmask_asmask,dilate_kernel_for_cb_mask,iterations = 1)     

            color_or_bw = cv2.bitwise_or(frame_bw, hsvmask, mask=hsvmask_asmask)   
        else:
            color_or_bw = cv2.bitwise_or(frame_bw, hsvmask)  

        # Do some morphyology to enhance the result
        frame_bw = cv2.morphologyEx(frame_bw, cv2.MORPH_OPEN, morph_open_kernel)
        color_or_bw = cv2.dilate(color_or_bw,dilate_kernel_for_cb_mask,iterations = 1)

        if isMarkerDebug:  
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_11color_or_bw.jpg'), color_or_bw)

        # Do contour extraction for the final segmentation result
        logger.info('Start finding contour....')
        image, contours, hierarchy =cv2.findContours(color_or_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        for cnt in contours : 
            cv2.fillPoly(color_or_bw, pts =[cnt], color=(255,255,255))

        if isMarkerDebug:  
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_12fillPoly.jpg'), color_or_bw)


        # find contour again after filling polygon
        image, contours, hierarchy =cv2.findContours(color_or_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

        frame_bw_color = cv2.cvtColor(color_or_bw, cv2.COLOR_GRAY2BGR)

        obj_id=0
        for cnt in contours : 
            color = colors[int(obj_id) % len(colors)]
            cv2.drawContours(frame_bw_color, cnt, -1, color, 5)
            obj_id=obj_id+1

        if isMarkerDebug:  
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_13findContours.jpg'), frame_bw_color)
        
        # Searching through every region selected to  
        # find the required polygon. 

        obj_id=0
        good_contours=[]
        approximg= img2.copy()
        minrectimg= img2.copy()
        hull = []

        # 0.25 is the ideal area to perimeter ratio for a square
        area_to_perimeter_ratio_lower_threshold = 0.25*(1-area_to_perimeter_ratio_threshold*2.5)
        area_to_perimeter_ratio_upper_threshold = 0.25*(1+area_to_perimeter_ratio_threshold*2.5)

        if i is 1:
            area_lower_threshold = area_in_px*(1-marker_area_variation)
            area_upper_threshold = area_in_px*(1+marker_area_variation)
            area_to_perimeter_ratio_lower_threshold = 0.25*(1-area_to_perimeter_ratio_threshold)
            area_to_perimeter_ratio_upper_threshold = 0.25*(1+area_to_perimeter_ratio_threshold)

        logger.info('Area lower threshold: %s', str(area_lower_threshold))
        logger.info('Area upper threshold: %s', str(area_upper_threshold))
        logger.info('area_to_perimeter_ratio lower threshold: %s', str(area_to_perimeter_ratio_lower_threshold))
        logger.info('area_to_perimeter_ratio upper threshold: %s', str(area_to_perimeter_ratio_upper_threshold))
        logger.info('Number of contour: %s', str(len(contours)))

        # fine tune approxPolyDP_epsilon to avoid making square to triangle
        # when square to triangle, 2r=>1.414r, so total decrease in perimeter = 0.59r
        max_epsilon = side_in_px*0.59*0.7
        logger.info('max_epsilon: %s', str(max_epsilon))

        approxPolyDP_epsilon = min(approxPolyDP_epsilon, max_epsilon)

        # main loop to process each contour candidate, will filter them by area, area_to_perimeter ratio, etc...
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
            if area > area_lower_threshold and area < area_upper_threshold and ((ww>hh*aspect_ratio_thd and i is 0) or i is 1):  

                # hull.append(cv2.convexHull(cnt, False))   
                # cv2.drawContours(approximg, hull, 0, color, 5)      
                approxPolyDP_epsilon = approxPolyDP_arclength_ratio*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,  approxPolyDP_epsilon, False) 

                # Assume it shape is close to parallelogram, so after polygon approximation will close to 4 sides
                if len(approx) > length_approx_threshold_low and len(approx) < length_approx_threshold_high:
                    approx = simplify_contour(cnt, 4)
                    perimeter = cv2.arcLength(approx,True)
                    ratio = math.sqrt(area)/(perimeter+1e-10)                
                    
                    if ratio > area_to_perimeter_ratio_lower_threshold and ratio < area_to_perimeter_ratio_upper_threshold:

                        cv2.drawContours(approximg, [cnt], 0, color2, 1) 
                        cv2.drawContours(minrectimg, [cnt], 0, color2, 1) 

                        logger.info('obj_id: %s center: ( %s, %s ) area: %s perimeter: %s len(approx): %s ratio: %s', str(obj_id), str(cX), str(cY), str(area), str(perimeter), str(len(approx)), str(ratio))       
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
                            
                        if IsUsingRefinedCorner:

                            approx_new = corner_refinement(i, obj_id, x, y, m, approx, imgg, imgg_color, mergedlineminthreshold[i], 
                                scalethd_for_intersected_lsd, RefinedCornerThreshold[i], sameline_thd, 
                                lines_merging_min_length, lines_merging_angle_diff_in_degree, lines_merging_extension_ratio,
                                lines_merging_maxgap, isMarkerDebug, logger, output_dir)

                            approx = np.asarray(approx_new)

                        result_contour_img[i].append(imgg)
                        result_contour_id[i].append(obj_id)
                        result_contour_cnt[i].append(approx)
                        result_contour_ROI[i].append((x, y, w, h))
                    
                    else:
                        logger.info('one contour rejected by area_to_perimeter_ratio, which ratio= %s _obj_id: %s center:( %s, %s ) area: %s ================', str(ratio), str(obj_id), str(cX), str(cY), str(area))
                    
                else:
                    logger.info('one contour rejected by len(approx), which len(approx)= %s _obj_id: %s center:( %s, %s ) area: %s ================', str(len(approx)), str(obj_id), str(cX), str(cY), str(area))

            obj_id=obj_id+1

        if isMarkerDebug:
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_14approxPolyDP.jpg'), approximg)
        logger.info('good contour number: %s', str(len(result_contour_id[i])))

        outputmask = np.zeros(color_or_bw.shape, dtype="uint8") 
        for cnt in good_contours : 
            cv2.drawContours(outputmask, [cnt], 0, (255,255,255), -1) 

        outputmask = cv2.dilate(outputmask,dilate_kernel_for_cb_mask,iterations = 2)
        outputmask = cv2.resize(outputmask, (0,0), fx=scaledownratio[i], fy=scaledownratio[i])

        if isMarkerDebug:
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_15outputmask.jpg'), outputmask)


        obj_id=0
        for cnt in good_contours : 
            color = colors[int(obj_id) % len(colors)]
        
            rect = cv2.minAreaRect(cnt) 
            box = cv2.boxPoints(rect) 
            box = np.int0(box)
            cv2.drawContours(minrectimg, [box], 0, color, 1)

            obj_id=obj_id+1

        if isMarkerDebug:
            cv2.imwrite(os.path.join(output_dir, 'i=' + str(i) + '_16detected.jpg'), minrectimg)


    logger.info('=====================================================================')
    logger.info('Matching stage')
    logger.info('Method: Area of hue vector')

    if len(result_contour_id[0]) > 0 and len(result_contour_id[1]) > 0:

        # After filtering marker contour candidate on both sides, need to match each of them
        matched_indices = marker_matching_by_hue_vector(result_contour_ROI, result_contour_id, lowerhsvlist, upperhsvlist, frame_hsv_original_copy, frame_bwh_original_copy, 
            frame_bwl_original_copy, result_contour_img, logger)

        # After matching individual markers, also need to determine the order of the 4 corners pt
        final_src_pts, final_dst_pts = determine_corner_point_order(matched_indices, result_contour_cnt, result_contour_id, logger)

        if final_src_pts is None:
            logger.warning('Corner point order determination fail!')
            return None, None

        else:
            if isMarkerDebug:
                output_matching_img(frame_img2, output_dir, final_src_pts, final_dst_pts)

            logger.info('%s pts are added to final RANSAC homography', str(len(final_dst_pts)))

            return final_dst_pts*scaledownratio[1],final_src_pts*scaledownratio[0]
    
    else:
        logger.warning('No marker contour candidate is found!')
        return None, None

def ComputeReprojectionErr1(pp1_gps, pp2, H_px_to_gps):

    """
    function to compute reprojection error of H_px_to_gps

    Note:
        N/A

    Args:
        pp1_gps (nparray): array that stores gps coordinate
        pp2 (nparray): array that stores pixel coordinate
        H_px_to_gps (nparray): homography matrix that maps from pixel to gps

    Returns:
        rms_err (double): root mean squared error of reprojection

    Raises:
        N/A
    
    """

    temp = np.array([pp2])
    reproject_pts_gps = cv.perspectiveTransform(temp, H_px_to_gps)
    
    rms_err = 0
    sum_of_data = 0
    for i in range(0, pp1_gps.shape[0]):
        _GEOD = pyproj.Geod(ellps='WGS84')
        _,_,d = _GEOD.inv(pp1_gps[i][1],pp1_gps[i][0],reproject_pts_gps[0][i][1],reproject_pts_gps[0][i][0]) 
        rms_err = rms_err + math.pow(d, 2)
        sum_of_data = sum_of_data + 1

    rms_err = math.sqrt(rms_err/sum_of_data)

    return rms_err

def ComputeReprojectionErr2(pp1_gps, pp2, H_px_to_px, scale_x, scale_y, img1_cols, img1_rows, min_x, min_y, max_x, max_y):

    """
    function to compute reprojection error of H_px_to_px

    Note:
        N/A

    Args:
        pp1_gps (nparray): array that stores gps coordinate
        pp2 (nparray): array that stores pixel coordinate
        H_px_to_px (nparray): homography matrix that maps from pixel to pixel
        scale_x (double): scale down factor in column direction
        scale_y (double): scale down factor in row direction
        img_cols (int): column size of orthophoto in original size
        img_rows (int): row size of orthophoto in original size
        min_x (double): min x value in gps coordinate
        min_y (double): min y value in gps coordinate
        max_x (double): max x value in gps coordinate
        max_y (double): max y value in gps coordinate

    Returns:
        rms_err (double): root mean squared error of reprojection

    Raises:
        N/A
    
    """

    temp = np.array([pp2])
    reproject_pts_px = cv.perspectiveTransform(temp, H_px_to_px)

    rms_err = 0
    sum_of_data = 0
    for i in range(reproject_pts_px.shape[1]):
        _GEOD = pyproj.Geod(ellps='WGS84')
        new_pt = ConvertPx2GPS(reproject_pts_px[0][i][0], reproject_pts_px[0][i][1], scale_x, scale_y, img1.shape[1], img1.shape[0], minx, miny, maxx, maxy)
        _,_,d = _GEOD.inv(pp1_gps[i][1],pp1_gps[i][0], new_pt[0][1], new_pt[0][0]) 
        rms_err = rms_err + math.pow(d, 2)
        sum_of_data = sum_of_data + 1
     
    rms_err = math.sqrt(rms_err/sum_of_data)

    return rms_err

def match_and_draw(desc1, desc2, kp1, kp2, resized_img1, resized_img2, mask1, mask2, img1_cols, img1_rows, scale_x, 
    scale_y, min_x, min_y, max_x, max_y, marker_pt_to_ASIFT_pt_ratio, SIFT_ratio, RANSAC_iteration_num, 
    is_need_marker, logger, is_debug, additional_points_pair_file, marker_config_file, img1_path, img2_path,
    orthomosaicscale, is_time_log, output_dir):
    
    """
    function to match key points, estimate homography matrix and output matching image

    Note:
        N/A

    Args:
        desc1 (opencv defined type): key point descriptors of orthophoto
        desc2 (opencv defined type): key point descriptors of perspective image
        kp1 (opencv defined type): key points of orthophoto
        kp2 (opencv defined type): key points of perspective image
        resized_img1 (nparray): scale down orthophoto
        resized_img2 (nparray): scale down perspective image
        img1_cols (int): column size of orthophoto in original size
        img1_rows (int): row size of orthophoto in original size
        scale_x (double): scale down factor in column direction
        scale_y (double): scale down factor in row direction
        min_x (double): min x value in gps coordinate
        min_y (double): min y value in gps coordinate
        max_x (double): max x value in gps coordinate
        max_y (double): max y value in gps coordinate
        marker_pt_to_ASIFT_pt_ratio (int): a integer ratio to tune the weighting between marker pt vs ASIFT pt in the final fitting
        SIFT_ratio (double): SIFT ratio between 1st and 2nd knn match to filter out bad matches
        RANSAC_iteration_num (int): Max iteration number for RANSAC homography
        is_need_marker (bool): Whether marker is used to assist calibration
        logger (python object): A logging handler
        is_debug (bool): A debug flag
        marker_config_file (string): path to the marker detection config file
        img1_path (string): image path to the orthomosaic image
        img2_path (string): image path to the perspective image
        orthomosaicscale (double): m/px scale for the orthomosaic image

        

    Returns:
        H_px_to_gps (nparray): homography matrix that maps from pixel to gps by single mapping (ryan's method)
        rms_err_H_px_to_gps (double): rms error of H_px_to_gps
        H_px_to_gps2 (nparray):  homography matrix that maps from pixel to pixel by double mapping (vincent's method)
        rms_err_H_px_to_gps2 (double): rms error of H_px_to_gps2
        H_px_to_px (nparray):  homography matrix that maps from pixel to pixel 
        rms_err_H_px_to_px (double): rms error of H_px_to_px
        match_img (nparray): image showing inlier matching pairs

    Raises:
        N/A
    
    """
    
    start = time.time()
    logger.info('Start to compute knn matches of the keypoints (it takes some time)....')
    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2

    logger.info('compute knn matches of the keypoints - complete')
    if (is_time_log): 
        logger.info('compute knn matches of the keypoints took %f seconds.',time.time() - start)

    start = time.time()
    logger.info('Filter matches by do SIFT ratio test...')
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches, SIFT_ratio) 
    logger.info('Filter matches by do SIFT ratio test - complete')
    if (is_time_log): 
        logger.info('Filter matches by do SIFT ratio test took %f seconds.',time.time() - start)

    marker_pt1 = None
    marker_pt2 = None
    if is_need_marker:
        # img1: ortho, img2, persepctive
        start = time.time()
        marker_pt1, marker_pt2 = detect_markers(marker_config_file, logger, img1_path, img2_path, is_debug, orthomosaicscale)
        logger.info('detect marker - complete')
        if (is_time_log): 
            logger.info('detect marker took %f seconds.',time.time() - start)

    gray1 = cv.cvtColor(resized_img1, cv.COLOR_BGR2GRAY)
    if not mask1 is None:
        gray1_masked = cv.bitwise_and(gray1, mask1)
        gray1 = cv2.addWeighted(gray1, 0.3, gray1_masked, 0.7, 0)
    gray2 = cv.cvtColor(resized_img2, cv.COLOR_BGR2GRAY)
    if not mask2 is None:
        gray2_masked = cv.bitwise_and(gray2, mask2)
        gray2 = cv2.addWeighted(gray2, 0.3, gray2_masked, 0.7, 0)

    logger.info('RANASC reprojection thd in px: %s', str(ransac_reprojection_threshold))

    number_of_asift_matches = len(p1)
    number_of_asift_inliers_matches = 0
    number_of_final_matches_pts = 0
    number_additional_points_pair = 0
    logger.info('number of matches: %s', str(number_of_asift_matches))

    pp1 = None
    pp2 = None

    if number_of_asift_matches >= 4 or is_need_marker and not marker_pt1 is None and number_of_asift_matches + len(marker_pt1) > 4:

        start = time.time()
        logger.info('Start to compute homography using RANSAC (it takes some time)...')

        # 1. inliers of ASIFT
        if number_of_asift_matches > 0:
            H, status = cv.findHomography(p2, p1, cv.RANSAC, ransac_reprojection_threshold, maxIters = RANSAC_iteration_num)
            kp_pairs_inliers = [kpp for kpp, flag in zip(kp_pairs, status) if flag]

            #just extract inliers
            pp1 = np.zeros(shape = (np.sum(status), 2))
            pp2 = np.zeros(shape = (np.sum(status), 2))

            j = 0
            for i in range(len(p1)):
                if (status[i] == 1):
                    pp1[j][0] = p1[i][0]
                    pp1[j][1] = p1[i][1]
                    pp2[j][0] = p2[i][0]
                    pp2[j][1] = p2[i][1]
                    j = j + 1

            number_of_asift_inliers_matches = str(len(pp1))
            logger.info('number of inliers of ASIFT matched points: %s', number_of_asift_inliers_matches)

        # 2. Artificial marker detection point pairs
        if is_need_marker:
            if marker_pt1 is None:
                logger.info('Marker detection fail! Continue to use ASIFT point for RANSAC homography')

            else:
                H, status = cv.findHomography(marker_pt2, marker_pt1, cv.RANSAC, (0.5*ransac_reprojection_threshold)/scale_x, maxIters = RANSAC_iteration_num)
                logger.info('marker RANSAC inliers: %s', str(cv.transpose(status)))

                logger.info('number of inliers of the marker points: %s', str(np.sum(status)))

                natural_kp_inliers_to_marker_pts_ratio = max(1, marker_pt_to_ASIFT_pt_ratio*int(len(pp1)/np.sum(status)))
                logger.info('natural kp inlier to marker pts ratio: %s', str(natural_kp_inliers_to_marker_pts_ratio))            

                for j in range(0, natural_kp_inliers_to_marker_pts_ratio):
                    for i in range(len(marker_pt1)):
                        if (status[i] == 1):
                            if pp1 is None and pp2 is None:
                                pp1 = np.zeros(shape = (1, 2))
                                pp2 = np.zeros(shape = (1, 2))
                                pp1[0][0] = marker_pt1[i][0][0]*scale_x
                                pp1[0][1] = marker_pt1[i][0][1]*scale_y
                                pp2[0][0] = marker_pt2[i][0][0]
                                pp2[0][1] = marker_pt2[i][0][1]

                            else:
                                pp1 = np.append(pp1, np.asarray([[marker_pt1[i][0][0]*scale_x, marker_pt1[i][0][1]*scale_y]]), axis = 0)   
                                pp2 = np.append(pp2, np.asarray([[marker_pt2[i][0][0], marker_pt2[i][0][1]]]), axis = 0) 
                
                logger.info('Total pts to fit: %s', str(len(pp1)))

        # 3. Additional points pair by manual pairing
        if not additional_points_pair_file == "":
            fs_read = cv2.FileStorage(additional_points_pair_file, cv2.FILE_STORAGE_READ)
            ortho_matrix = fs_read.getNode("ortho_matrix").mat()
            perspective_matrix = fs_read.getNode("perspective_matrix").mat()
            fs_read.release()

            logger.info('%s additional manual points pair will be added to the final homography fitting', str(ortho_matrix.shape[0]))
            for idx in range(0, ortho_matrix.shape[0]):
                new_pp1 = np.array([[ortho_matrix[idx][1]*scale_x, ortho_matrix[idx][2]*scale_y]])
                new_pp2 = np.array([[perspective_matrix[idx][1], perspective_matrix[idx][2]]])

                pp1 = np.append(pp1, new_pp1, axis = 0)
                pp2 = np.append(pp2, new_pp2, axis = 0)

                logger.info('ortho %s %s, perspective %s %s', str(new_pp1[0][0]), str(new_pp1[0][1]), str(new_pp2[0][0]), str(new_pp2[0][1]))
                number_additional_points_pair = number_additional_points_pair + 1


        #Sum of 1 and 2 and 3, convert final matching pts from px to gps
        number_of_final_matches_pts = len(pp1)
        pp1_gps = np.zeros(shape = (pp1.shape[0], pp1.shape[1]))
        for i in range(len(pp1)):
            temp = ConvertPx2GPS(pp1[i][0], pp1[i][1], scale_x, scale_y, img1_cols, img1_rows, min_x, min_y, max_x, max_y)   
            pp1_gps[i][0] = temp[0][0]
            pp1_gps[i][1] = temp[0][1]

            
        H_px_to_gps, status = cv.findHomography(pp2, pp1_gps, 0)

        H_px_to_px, status = cv.findHomography(pp2, pp1, 0)

        #convert H_px_to_px to H_px_to_gps2
        scale_mat = np.array([[ 1/scale_x, 0 , 0], 
                              [ 0, 1/scale_y , 0], 
                              [ 0, 0 ,1]])
        cvt_gps_mat = np.array([[ 0, (min_x - max_x)/img1_rows, max_x], 
                                [ (max_y - min_y)/img1_cols, 0, min_y], 
                                [ 0, 0 ,1]])  

        H_px_to_gps2 = cvt_gps_mat.dot(scale_mat.dot(H_px_to_px))

        logger.info('Compute homography - complete')
        if (is_time_log): 
            logger.info('Compute homography took %f seconds.',time.time() - start)            

        # The drawing part
        start = time.time()
        logger.info('Start to draw the results...')
        keypt_img = explore_match(gray1, gray2, kp_pairs, True, None, None, H_px_to_px, logger, None, False, 0, None)
        match_img = explore_match(gray1, gray2, kp_pairs, False, None, None, H_px_to_px, None, output_dir, True, 0, None)
        match_img_inliers = explore_match(gray1, gray2, kp_pairs_inliers, False, None, None, H_px_to_px, None, None, False, 0, None)
        match_img_final_matchpts = explore_match(gray1, gray2, None, False, pp1, pp2, H_px_to_px, None, output_dir, True, number_additional_points_pair, pp1_gps)
        logger.info('Result drawing - complete')
        if (is_time_log): 
            logger.info('Result drawing took %f seconds.',time.time() - start)        

    else:
        H_px_to_gps, H_px_to_gps2, H_px_to_px, match_img, match_img_inliers = None, None, None, None

    #compute reprojection error
    start = time.time()
    logger.info('Start to compute reprojection error...')    
    if H_px_to_gps is not None:
        rms_err_H_px_to_gps = ComputeReprojectionErr1(pp1_gps, pp2, H_px_to_gps)
    else:
        rms_err_H_px_to_gps = None
    if H_px_to_gps2 is not None:    
        rms_err_H_px_to_px = ComputeReprojectionErr2(pp1_gps, pp2, H_px_to_px, scale_x, scale_y, img1_cols, img1_rows, min_x, min_y, max_x, max_y)
        rms_err_H_px_to_gps2 = ComputeReprojectionErr1(pp1_gps, pp2, H_px_to_gps2)
    else:
        rms_err_H_px_to_px = None
        rms_err_H_px_to_gps2 = None

    logger.info('Compute reprojection error - complete')
    if (is_time_log): 
        logger.info('Compute reprojection error took %f seconds.',time.time() - start) 

    return H_px_to_gps, rms_err_H_px_to_gps, H_px_to_gps2, rms_err_H_px_to_gps2, H_px_to_px, rms_err_H_px_to_px, keypt_img, match_img , match_img_inliers, match_img_final_matchpts, number_of_asift_matches, number_of_asift_inliers_matches, number_of_final_matches_pts

if __name__ == '__main__':

    #load script path
    script_path = os.path.dirname(os.path.realpath(sys.argv[0])) 

    #load runtime config file
    runtime_config_read = cv.FileStorage("match_runtime_config.yml", cv.FILE_STORAGE_READ) 
    match_config_file_path = runtime_config_read.getNode("match_config_file_path").string()
    log_config_file_path = runtime_config_read.getNode("log_config_file_path").string()
    camera_id = runtime_config_read.getNode("camera_id").string()

    #load log config file 
    loop_start_time = time.time()
    start = loop_start_time

    try:
        log_config_read = cv.FileStorage(log_config_file_path, cv.FILE_STORAGE_READ) 
    except:
        logger.error("fail to load log config file")
        exit(1)
        
    logging_file_name = log_config_read.getNode("logging_file_name").string()
    
    if (logging_file_name == ""):
        print("matching logging file name is empty.")
        exit(1)

    max_log_size_in_mb = log_config_read.getNode("max_log_file_size_in_mb").real()
    if (max_log_size_in_mb <= 0):
        print("max log size in mb is zero or negative")
        exit(1)

    logging_format = log_config_read.getNode("logging_format").string()
    if (logging_format == ""):
        print("logging format is empty")
        exit(1)

    is_debug = log_config_read.getNode("is_debug").string().lower()
    if is_debug == "true":
        is_debug = True
    elif is_debug == "false":
        is_debug = False
    else:
        print("incorrect is_debug")
        exit(1)

    output_dir = None
    if (is_debug):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = "./Matching_Debug_"+str(st) +"/"
        os.mkdir(output_dir)

    is_print_to_console = log_config_read.getNode("is_print_to_console").string().lower()
    if is_print_to_console == "true":
        is_print_to_console = True
    elif is_print_to_console == "false":
        is_print_to_console = False
    else:
        print("incorrect is_print_to_console")
        exit(1)

    is_time_log = log_config_read.getNode("is_time_log").string().lower()
    if is_time_log == "true":
        is_time_log = True
    elif is_time_log == "false":
        is_time_log = False
    else:
        print("incorrect is_time_log")
        exit(1)

    logger = logging.getLogger('MyLogger')
    log_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    logger.setLevel(logging.DEBUG)

    stdouthandler = logging.StreamHandler(sys.stdout)
    stdouthandler.setLevel(logging.INFO)
    stdoutformatter = log_formatter
    stdouthandler.setFormatter(stdoutformatter)

    my_handler = RotatingFileHandler(logging_file_name, mode=logging_format, maxBytes=max_log_size_in_mb*1024*1024, 
                                    backupCount=2, encoding=None, delay=False)
    my_handler.setFormatter(log_formatter)
    my_handler.setLevel(logging.DEBUG)
    logger.addHandler(my_handler)

    if (is_print_to_console): 
        logger.addHandler(stdouthandler)

    logger.info('================ start of logging match.py ================')
    logger.info('load logging config - complete')

    if (is_time_log): 
        logger.info('load log config took %f seconds.',time.time() - start)


    logger.info("[========SUMMARY OF MATCH RUNTIME CONFIG========]")
    logger.info("match_config_file_path: %s", match_config_file_path)
    logger.info("log_config_file_path: %s", log_config_file_path)
    logger.info("camera_id: %s", camera_id)
    logger.info("[========END SUMMARY OF MATCH RUNTIME CONFIG========]")

    #load match config file
    start = time.time()
    try:
        match_config_read = cv.FileStorage(match_config_file_path, cv.FILE_STORAGE_READ)
    except:
        logger.error("fail to load match config file")
        exit(1)

    #load the weighting between marker points and ASIFT points
    marker_pt_to_ASIFT_pt_ratio = int(match_config_read.getNode("marker_pt_to_ASIFT_pt_ratio").real())
    if (marker_pt_to_ASIFT_pt_ratio <= 0):
        logger.error("incorrect marker_pt_to_ASIFT_pt_ratio, should be greater than 0")
        exit(1)

    #load max threads used
    max_threads_used = int(match_config_read.getNode("max_threads_used").real())
    if (max_threads_used <= 0):
        logger.error("incorrect max_threads_used, should be greater than 0")
        exit(1)

    #load rms matching error 
    max_rms_err_thres_in_m = int(match_config_read.getNode("max_rms_err_thres_in_m").real())
    if (max_rms_err_thres_in_m <= 0):
        logger.error("incorrect max_rms_err_thres_in_m, should be greater than 0")
        exit(1)

    #load the scale ratio for orthophoto
    orthophoto_img_scale = match_config_read.getNode("orthophoto_img_scale").real()
    if (orthophoto_img_scale <= 0 or orthophoto_img_scale > 1):
        logger.error("incorrect orthophoto_img_scale, should be within (0 - 1]")
        exit(1)

    #load SIFT ratio
    SIFT_ratio = match_config_read.getNode("SIFT_ratio").real()
    if (SIFT_ratio <= 0 or SIFT_ratio >= 1):
        logger.error("incorrect SIFT_ratio, should be within (0 - 1)")
        exit(1)

    #load RANSAC_iteration_num
    RANSAC_iteration_num = int(match_config_read.getNode("RANSAC_iteration_num").real())
    if (RANSAC_iteration_num <= 0):
        logger.error("incorrect ransac interation number, should be greater than 0")
        exit(1)

    #load RANSAC reprojection threshold
    RANSAC_phy_reproject_thres = match_config_read.getNode("RANSAC_phy_reprojection_thres_in_m").real()
    if (RANSAC_phy_reproject_thres <= 0):
        logger.error("incorrect RANSAC_phy_reprojection_thres_in_m, should be greater than 0")
        exit(1)

    #load if distortion correction is required
    IsNeedDistortionCorrection = (match_config_read.getNode("is_need_distortion_corr").string()).lower() 
    if IsNeedDistortionCorrection == "true":
        IsNeedDistortionCorrection = True
    elif IsNeedDistortionCorrection == "false":
        IsNeedDistortionCorrection = False
    else:
        logger.error("incorrect is_need_distortion_corr, should be either true or false")
        exit(1)

    #load intrinsic_mat and dist_coef if distortion correction is required
    if IsNeedDistortionCorrection == True:
        #load file path 
        intrinsic_and_distortion_coeff_path = match_config_read.getNode("intrinsic_and_distortion_coeff_path").string()
        if (intrinsic_and_distortion_coeff_path == ""):
            logger.error("intrinsic matrix and distortion file name is empty.")
            exit(1)
        try:
            intrinsic_and_distortion_coeff_read = cv.FileStorage(intrinsic_and_distortion_coeff_path, cv.FILE_STORAGE_READ)
        except:
            logger.error("fail to load intrinsic and distortion file")
            exit(1)

        #load intrinsic matrix
        cam_mat = intrinsic_and_distortion_coeff_read.getNode("intrinsic_mat").mat() 
        if(cam_mat is None):
            logger.error("intrinsic matrix is not available")
            exit(1)   
        #load distortion coefficient
        dist_coef = intrinsic_and_distortion_coeff_read.getNode("distortion_coeff").mat()
        if(dist_coef is None):
            logger.error("distortion coefficients are not available")
            exit(1)   

    #load if marker detection is required
    IsNeedMarkerDetection = (match_config_read.getNode("is_need_marker").string()).lower() 
    if IsNeedMarkerDetection == "true":
        IsNeedMarkerDetection = True
    elif IsNeedMarkerDetection == "false":
        IsNeedMarkerDetection = False
    else:
        logger.error("incorrect is_need_marker, should be either true or false")
        exit(1)   

    logger.info('load match config - complete')

    if (is_time_log): 
        logger.info('load match config took %f seconds.',time.time() - start)

    #load img and geotiff
    start = time.time()
    orthophoto_img_path = match_config_read.getNode("orthophoto_img_path").string() 
    perspective_img_path = match_config_read.getNode("perspective_img_path").string()
    geotiff_path = match_config_read.getNode("geotiff_path").string()

    if orthophoto_img_path == "" or perspective_img_path == "" or geotiff_path == "":
        logger.error("empty image/geotiff path")
        exit(1)

    img1 = cv.imread(orthophoto_img_path)
    img2 = cv.imread(perspective_img_path)

    if img1 is None:
        logger.error('Failed to load orthophoto: %s', orthophoto_img_path)
        exit(1)

    if img2 is None:
        logger.error('Failed to load perspective image: %s', perspective_img_path)
        exit(1)

    ds = gdal.Open(geotiff_path)

    logger.info('load perspective image and geotiff - complete')
    if (is_time_log): 
        logger.info('load perspective image and geotiff took %f seconds.',time.time() - start)

    #load mask
    start = time.time()
    orthophoto_mask_path = match_config_read.getNode("orthophoto_mask_path").string()
    perspective_mask_path = match_config_read.getNode("perspective_mask_path").string()
    marker_config_file = match_config_read.getNode("marker_config_file").string()
    additional_points_pair_file = match_config_read.getNode("additional_points_pair_file").string()

    if orthophoto_mask_path == "":
        mask1 = None
    else:
        mask1 = cv.imread(orthophoto_mask_path, 0)
        if mask1 is None:
            logger.error("cannot load orthophoto_mask")
            exit(1)

    if perspective_mask_path == "":
        mask2 = None
    else:
        mask2 = cv.imread(perspective_mask_path, 0)
        if mask2 is None:
            logger.error("cannot load perspective_mask")
            exit(1)

    logger.info('load mask - complete')
    if (is_time_log): 
        logger.info('load mask took %f seconds.',time.time() - start)

    #undistort image
    start = time.time()
    if IsNeedDistortionCorrection == True:
        new_cam_mat, new_roi = cv.getOptimalNewCameraMatrix(cam_mat, dist_coef, (img2.shape[1],img2.shape[0]), 1, (img2.shape[1],img2.shape[0]))
        undist_img = cv.undistort(img2, cam_mat, dist_coef, None, new_cam_mat)
        (x,y,w,h) = new_roi
        if (w == 0 or h == 0):
            logger.error("cannot undistort image, probably due to incorrect distortion coefficient or intrinsic matrix")
            exit(1)
        img2 = undist_img[y:y+h,x:x+w].copy()

    logger.info('undistort perspective image - complete')
    if (is_time_log): 
        logger.info('undistort perspective image took %f seconds.',time.time() - start)

    #set detector
    start = time.time()
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift-flann')  
    detector, matcher = init_feature(feature_name)

    if detector is None:
        logger.error('unknown feature: %s', feature_name)
        sys.exit(1)

    logger.info('set detector - complete')
    if (is_time_log): 
        logger.info('set detector took %f seconds.',time.time() - start)

    logger.info('Size of ortho before scaling: %s %s', str(img1.shape[1]), str(img1.shape[0]))
    logger.info('Size of perspective before scaling: %s %s', str(img2.shape[1]), str(img2.shape[0]))

    #scale down image (any idea about this ad-hoc logic? any improvement can be made?)
    start = time.time()
    # if np.max([img1.shape[0], img1.shape[1]]) < 5000:
    #     scale_x = 1.0
    #     scale_y = 1.0
    # elif np.max([img1.shape[0], img1.shape[1]]) >= 5000 and np.max([img1.shape[0], img1.shape[1]]) < 10000:    
    #     scale_x = 0.7
    #     scale_y = 0.7
    # else:
    #     scale_x = 0.6
    #     scale_y = 0.6

    scale_x = orthophoto_img_scale
    scale_y = orthophoto_img_scale

    logger.info('Imply scale ratio %f on to the orthomosaic image', scale_x)
    resized_img1 = cv.resize(img1, (0, 0), None, scale_x, scale_y)
    resized_img2 = img2.copy()     

    logger.info('Size of ortho after scaling: %s %s', str(resized_img1.shape[1]), str(resized_img1.shape[0]))
    logger.info('Size of perspective after scaling: %s %s', str(resized_img2.shape[1]), str(resized_img2.shape[0]))    

    logger.info('scale down geotiff - complete')
    if (is_time_log): 
        logger.info('scale down geotiff took %f seconds.',time.time() - start)

    #scale down geotiff mask
    start = time.time()
    if (mask1 is not None):
        resized_mask1 = cv.resize(mask1, (0, 0), None, scale_x, scale_y)
    else:
        resized_mask1 = mask1

    logger.info('scale down mask - complete')
    if (is_time_log): 
        logger.info('scale down mask took %f seconds.',time.time() - start)

    #compute ransac_reprojection_threshold 
    logger.info('Parse the .tif by using gdalinfo to get the 4 corners gps information...')
    start = time.time()
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()

    #ulx, xres, xskew, uly, yskew, yres  = ds.GetGeoTransform()
    #print("all: ", ulx, xres, xskew, uly, yskew, yres)

    miny = gt[0]
    minx = gt[3] + width*gt[4] + height*gt[5] 
    maxy = gt[0] + width*gt[1] + height*gt[2]
    maxx = gt[3] 

    logger.info('minx in geotiff(gdalinfo): %s', str(minx))
    logger.info('miny in geotiff(gdalinfo): %s', str(miny))
    logger.info('maxx in geotiff(gdalinfo): %s', str(maxx))
    logger.info('maxy in geotiff(gdalinfo): %s', str(maxy))

    pt1 = ConvertPx2GPS(1.0, 1.0, scale_x, scale_y, img1.shape[1], img1.shape[0], minx, miny, maxx, maxy)
    pt2 = ConvertPx2GPS(2.0, 1.0, scale_x, scale_y, img1.shape[1], img1.shape[0], minx, miny, maxx, maxy)

    logger.info('Use 2 adjacent pixel points to estimate the scale ratio between pixel to physical distance...')
    logger.info('pt1: px(1, 1): %s %s', str(pt1[0][0]), str(pt1[0][1]))
    logger.info('pt2: px(2, 1): %s %s', str(pt2[0][0]), str(pt2[0][1]))

    temp = pyproj.Geod(ellps='WGS84')   
    _,_,d1 = temp.inv(pt1[0][1],pt1[0][0],pt2[0][1],pt2[0][0]) 

    logger.info('physical distance of pt1 and pt2 under the geotiff (in m): %s', str(d1))

    if math.isnan(d1):
        logger.info('orthomosaicscale re-estimated is nan!! to try parse content in gdalinfo directly to try to get lat lon')
        CornerLats, CornerLons = GetCornerCoordinates(geotiff_path)

        miny = CornerLons[0]
        minx = CornerLats[0]
        maxy = CornerLons[3]
        maxx = CornerLats[3]

        if(math.fabs(CornerLons[0]-CornerLons[1])>1e-5 or math.fabs(CornerLons[2]-CornerLons[3])>1e-5 or
            math.fabs(CornerLats[0]-CornerLats[2])>1e-5 or math.fabs(CornerLats[1]-CornerLats[3])>1e-5):
            logger.info('Check if the geotiff is aligned to true north!!! Discover gps difference in data at same axis!')

        logger.info('new minx in geotiff(gdalinfo): %s', str(minx))
        logger.info('new miny in geotiff(gdalinfo): %s', str(miny))
        logger.info('new maxx in geotiff(gdalinfo): %s', str(maxx))
        logger.info('new maxy in geotiff(gdalinfo): %s', str(maxy))

        pt1 = ConvertPx2GPS(1.0, 1.0, scale_x, scale_y, img1.shape[1], img1.shape[0], minx, miny, maxx, maxy)
        pt2 = ConvertPx2GPS(2.0, 1.0, scale_x, scale_y, img1.shape[1], img1.shape[0], minx, miny, maxx, maxy)

        logger.info('new pt1: px(1, 1): %s %s', str(pt1[0][0]), str(pt1[0][1]))
        logger.info('new pt2: px(2, 1): %s %s', str(pt2[0][0]), str(pt2[0][1]))
    
        _,_,d1 = temp.inv(pt1[0][1],pt1[0][0],pt2[0][1],pt2[0][0]) 
        logger.info('physical distance of pt1 and pt2 under the geotiff (in m): %s', str(d1))


    if math.isnan(d1):
        logger.info('orthomosaicscale re-estimated is nan again!! set ransac_reprojection_threshold to 100 to make the program continue')
        ransac_reprojection_threshold = 100
    else:
        logger.info('orthomosaicscale re-estimated: %s', str(d1*scale_x))
        ransac_reprojection_threshold = RANSAC_phy_reproject_thres/d1*scale_x
    
    logger.info('compute ransac reprojection threshold - complete')
    if (is_time_log): 
        logger.info('compute ransac reprojection threshold took %f seconds.',time.time() - start)

    #define number of cpu used
    start = time.time()
    if (cv.getNumberOfCPUs() > max_threads_used):
        num_of_cpu = max_threads_used
    else:
        num_of_cpu = cv.getNumberOfCPUs()

    logger.info('define number of parallel threads - complete')
    if (is_time_log): 
        logger.info('define number of parallel threads took %f seconds.',time.time() - start)

    logger.info("[========SUMMARY OF MATCH CONFIG========]")
    logger.info("orthophoto_img_path: %s", orthophoto_img_path)
    logger.info("orthophoto_mask_path: %s", orthophoto_mask_path)
    logger.info("perspective_img_path: %s", perspective_img_path)
    logger.info("perspective_mask_path: %s", perspective_mask_path)
    logger.info("IsNeedDistortionCorrection: %s", IsNeedDistortionCorrection)
    logger.info("IsNeedMarkerDetection: %s", IsNeedMarkerDetection)
    logger.info("marker_config_file: %s", marker_config_file)
    logger.info("additional_points_pair_file: %s", additional_points_pair_file)
    logger.info("marker_pt_to_ASIFT_pt_ratio: %s", marker_pt_to_ASIFT_pt_ratio)
    # logger.info("scale_x: %f", scale_x)
    # logger.info("scale_y: %f", scale_y)
    logger.info("orthophoto_img_scale: %f", orthophoto_img_scale)
    logger.info("SIFT_ratio: %f", SIFT_ratio)
    logger.info("RANSAC_iteration_num: %d", RANSAC_iteration_num)
    logger.info("RANSAC_phy_reproject_thres: %f", RANSAC_phy_reproject_thres)
    logger.info("ransac_reprojection_threshold_in_px: %f", ransac_reprojection_threshold)
    logger.info("max_rms_err_thres_in_m: %f", max_rms_err_thres_in_m)
    logger.info("number of parallel threads: %d", num_of_cpu)
    logger.info("ASIFT feature_name: %s", feature_name)
    logger.info("[========END SUMMARY OF MATCH CONFIG========]")

    #compute matching
    start = time.time()

    logger.info('Start affine detect of orthomosaic image (it takes some time)...')
    logger.info('(Use htop to monitor the usage of RAM, if the process got killed, use less number of parallel threads)')
    kp1, desc1 = affine_detect(detector, resized_img1, resized_mask1, pool=ThreadPool(processes = num_of_cpu))
    logger.info('Start affine detect of perspective image (it takes some time)...')
    kp2, desc2 = affine_detect(detector, resized_img2, mask2, pool=ThreadPool(processes = num_of_cpu))

    # # Initiate SIFT detector
    # sift = cv.xfeatures2d.SIFT_create()

    # # find the keypoints and descriptors with SIFT
    # kp1, desc1 = sift.detectAndCompute(resized_img1,mask1)
    # kp2, desc2 = sift.detectAndCompute(resized_img2,mask2)

    logger.info('ASIFT: ortho - %d features, perspective - %d features', len(kp1), len(kp2))
    logger.info('compute key points - complete')
    if (is_time_log): 
        logger.info('compute key points took %f seconds.',time.time() - start)

    #After obtaining result of asift matching, do some preprocessing (e.g. marker/additional point pairs) and compute homography matrix
    H_px_to_gps, rms_err_H_px_to_gps, H_px_to_gps2, rms_err_H_px_to_gps2, H_px_to_px, rms_err_H_px_to_px, \
        keypt_img, match_img , match_img_inliers, match_img_final_matchpts, number_of_asift_matches, \
        number_of_asift_inliers_matches, number_of_final_matches_pts = match_and_draw(desc1, desc2, kp1, kp2, \
        resized_img1, resized_img2, resized_mask1, mask2, img1.shape[1], img1.shape[0], scale_x, scale_y, minx, miny, maxx, maxy, \
        marker_pt_to_ASIFT_pt_ratio, SIFT_ratio, RANSAC_iteration_num, IsNeedMarkerDetection, logger, is_debug, \
            additional_points_pair_file, marker_config_file, orthophoto_img_path, perspective_img_path, d1*scale_x, is_time_log, output_dir)

    #check if estimation of homography matrix is successful
    if H_px_to_gps is None or H_px_to_gps2 is None:
        logger.error("cannot estimate homography matrix due to insufficient key point pairs")
        exit(1)

    logger.info('rms error of H_px_to_gps(direct mapping): %f m', rms_err_H_px_to_gps)
    logger.info('rms error of H_px_to_gps2(double mapping): %f m', rms_err_H_px_to_gps2)


    #save result
    start = time.time()

    # Save as .png is more clear than .jpg
    if keypt_img is not None and is_debug:
        cv.imwrite(os.path.join(output_dir, "ASIFT_matching_all_pairs_only_keypoints.png"), keypt_img)

    if match_img is not None and is_debug:
        cv.imwrite(os.path.join(output_dir, "ASIFT_matching_all_pairs_(" + str(number_of_asift_matches) + ").png"), match_img)

    if match_img_inliers is not None and is_debug:
        cv.imwrite(os.path.join(output_dir, "ASIFT_matching_inliers_pairs_(" + str(number_of_asift_inliers_matches) + ").png"), match_img_inliers)

    if match_img_final_matchpts is not None and is_debug:
        cv.imwrite(os.path.join(output_dir, "Matching_final_pairs_(" + str(number_of_final_matches_pts) + ").png"), match_img_final_matchpts)    

    if (rms_err_H_px_to_gps > max_rms_err_thres_in_m or rms_err_H_px_to_gps2 > max_rms_err_thres_in_m):
        logger.error("fail to map pixel to gps due to large rms error, no homography file is produced, please change config setting and re-calibrate again!")
        exit(1)

    if (is_debug):
        result_file = cv.FileStorage(os.path.join(output_dir, "eval_config.yml"), cv.FILE_STORAGE_WRITE)
        result_file.write("orthophoto_img_path", orthophoto_img_path)
        result_file.write("perspective_img_path", perspective_img_path)

        if (IsNeedDistortionCorrection == True):
            result_file.write("is_need_distortion_corr", "True")  
            result_file.write("intrinsic_mat", cam_mat)
            result_file.write("distortion_coeff", dist_coef)
        else:
            result_file.write("is_need_distortion_corr", "False")  

        result_file.write("homography_mat_px_to_gps", H_px_to_gps)
        result_file.write("homography_mat_px_to_gps2", H_px_to_gps2)
        result_file.write("homography_mat_px_to_px", H_px_to_px)
        result_file.write("scale_x", scale_x)
        result_file.write("scale_y", scale_y)
        result_file.write("min_x", minx)
        result_file.write("min_y", miny)
        result_file.write("max_x", maxx)
        result_file.write("max_y", maxy)        
        result_file.release()

    final_homography_output_file_name = "homography.yml"
    if not camera_id == "":
        final_homography_output_file_name = "homography_" + camera_id + ".yml"

    logger.info('Homography file is outout at: %s', os.path.join(output_dir, final_homography_output_file_name))
    result_file = cv.FileStorage(os.path.join(output_dir, final_homography_output_file_name), cv.FILE_STORAGE_WRITE)
    if rms_err_H_px_to_gps < rms_err_H_px_to_gps2:
        result_file.write("homography_matrix", H_px_to_gps)   
        logger.info('rms_err_H_px_to_gps < rms_err_H_px_to_gps2, use the result of direct mapping') 
    else:
        result_file.write("homography_matrix", H_px_to_gps2)    
        logger.info('rms_err_H_px_to_gps >= rms_err_H_px_to_gps2, use the result of double mapping') 

    result_file.write("number_of_asift_matches", str(number_of_asift_matches))
    result_file.write("number_of_asift_inliers_matches", str(number_of_asift_inliers_matches))
    result_file.write("number_of_final_matches_pts", str(number_of_final_matches_pts))
    result_file.write("rms error of H_px_to_gps - direct mapping", str(rms_err_H_px_to_gps))
    result_file.write("rms error of H_px_to_gps2 - double mapping", str(rms_err_H_px_to_gps2))
    result_file.write("total feature in ortho img", str(len(kp1)))
    result_file.write("total feature in perspective img", str(len(kp2)))
    result_file.write("orthophoto_img_path", str(orthophoto_img_path))
    result_file.write("perspective_img_path", str(perspective_img_path))
    result_file.write("orthophoto_mask_path", str(orthophoto_mask_path))
    result_file.write("perspective_mask_path", str(perspective_mask_path))
    result_file.write("geotiff_path", str(geotiff_path))
    result_file.write("orthophoto_img_scale", str(orthophoto_img_scale))
    result_file.write("RANSAC_iteration_num", str(RANSAC_iteration_num))
    result_file.write("SIFT_ratio", str(SIFT_ratio))
    result_file.write("RANSAC_phy_reproject_thres_in_m", str(RANSAC_phy_reproject_thres))
    result_file.write("ransac_reprojection_threshold_in_px", str(ransac_reprojection_threshold))
    result_file.write("max_rms_err_thres_in_m", str(max_rms_err_thres_in_m))
    result_file.write("additional_points_pair_file", str(additional_points_pair_file))

    if not additional_points_pair_file == "":
        fs_read = cv2.FileStorage(additional_points_pair_file, cv2.FILE_STORAGE_READ)
        ortho_matrix = fs_read.getNode("ortho_matrix").mat()
        perspective_matrix = fs_read.getNode("perspective_matrix").mat()
        fs_read.release()

        result_file.write("additional_points_pair_ortho_matrix", ortho_matrix)        
        result_file.write("additional_points_pair_perspective_matrix", perspective_matrix)      

    result_file.release()

    logger.info('save result - complete')
    if (is_time_log): 
        loop_end_time = time.time()
        logger.info('save result took %f seconds.',loop_end_time - start)
        logger.info('one loop took %f seconds.', loop_end_time - loop_start_time)
    logger.info('================ end of logging match.py ================')

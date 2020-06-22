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

[Evaluation module]

Module to evaluate the accuracy performance of the calibrated homography over a set of pre-defined test points
"""

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import os
import pyproj
import sys, getopt
import time, datetime
import logging
import logging.handlers
from logging.handlers import RotatingFileHandler

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

if __name__ == '__main__':
    
    #load testing data
    args = sys.argv[1:]
    
    if len(args) is not 3:
        print('incorrect number of args. e.g. python3 eval.py [test_config].yml eval_config.yml log_config.yml')
        exit(1)

    testing_file_path = args[0]

    if (testing_file_path == ""):
        print("testing file path is empty")
        exit(1)

    eval_config_file_path = args[1]

    if (eval_config_file_path == ""):
        print("evaluation config file path is empty")
        exit(1)

    log_config_file_path = args[2]

    if (log_config_file_path == ""):
        print("log config file path is empty")
        exit(1)

    #load log config
    loop_start_time = time.time()
    start = loop_start_time
    try:
        log_config_read = cv.FileStorage(log_config_file_path, cv.FILE_STORAGE_READ) 
    except:
        print("fail to load log config file")
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

    logger.info('================ start of logging eval.py ================')
    logger.info('load logging config - complete')
    if (is_time_log): 
        logger.info('load log config took %f seconds.',time.time() - start)

    #load test data
    start = time.time()
    try:
        fs_read = cv.FileStorage(testing_file_path, cv.FILE_STORAGE_READ)
    except:
        logger.error("fail to load test data")
        exit(1)

    gps_matrix = fs_read.getNode("gps_matrix").mat()
    test_pt_matrix = fs_read.getNode("cross_subpixel_matrix").mat()

    if gps_matrix is None or test_pt_matrix is None:
        logger.error("testing data is not available")
        exit(1)

    logger.info('load test data - complete')
    if (is_time_log): 
        logger.info('load test data took %f seconds.',time.time() - start)

    #load evalation config
    start = time.time()  
    try:
        fs_read = cv.FileStorage(eval_config_file_path, cv.FILE_STORAGE_READ)
    except:
        logger.error("fail to load evaluation config")
        exit(1)

    #load image path
    orthophoto_img_path = fs_read.getNode("orthophoto_img_path").string() 
    perspective_img_path = fs_read.getNode("perspective_img_path").string()

    if orthophoto_img_path == "" or perspective_img_path == "":
        logger.error("empty image/geotiff path")
        exit(1)

    #load image
    img1 = cv.imread(orthophoto_img_path)
    img2 = cv.imread(perspective_img_path)

    if img1 is None:
        logger.error('Failed to load orthophoto: %s', orthophoto_img_path)
        exit(1)

    if img2 is None:
        logger.error('Failed to load perspective image: %s', perspective_img_path)
        exit(1)

    #load distortion correction setting
    IsNeedDistortionCorrection = (fs_read.getNode("is_need_distortion_corr").string()).lower() 

    if IsNeedDistortionCorrection == "true":
        IsNeedDistortionCorrection = True
    elif IsNeedDistortionCorrection == "false":
        IsNeedDistortionCorrection = False
    else:
        logger.error("incorrect is_need_distortion_corr, should either true or false")
        exit(1)

    logger.info('load evaluation config - complete')
    if (is_time_log): 
        logger.info('load evaluation config took %f seconds.',time.time() - start)

    #undistort image if required
    start = time.time()
    if IsNeedDistortionCorrection == True:
        #load intrinsic matrix
        cam_mat = fs_read.getNode("intrinsic_mat").mat() 

        if cam_mat is None:
            logger.error("intrinsic matrix is not available")
            exit(1)

        #load distortion coefficient
        dist_coef = fs_read.getNode("distortion_coeff").mat()

        if dist_coef is None:
            logger.error("distortion coefficients are not available")
            exit(1)

        #undistort image
        new_cam_mat, new_roi = cv.getOptimalNewCameraMatrix(cam_mat, dist_coef, (img2.shape[1],img2.shape[0]), 1, (img2.shape[1],img2.shape[0]))
        undist_img = cv.undistort(img2, cam_mat, dist_coef, None, new_cam_mat)
        (x,y,w,h) = new_roi
        if (w == 0 or h == 0):
            logger.error("cannot undistort image, probably due to incorrect intrinsic matrix or distortion coefficients")
            exit(1)
        img2 = undist_img[y:y+h,x:x+w].copy()      

    logger.info('undistort image - complete')
    if (is_time_log): 
        logger.info('undistort image took %f seconds.',time.time() - start)

    #load mapping setting
    start = time.time()
    H_px_to_gps = fs_read.getNode("homography_mat_px_to_gps").mat()
    H_px_to_gps2 = fs_read.getNode("homography_mat_px_to_gps2").mat()
    H_px_to_px = fs_read.getNode("homography_mat_px_to_px").mat()
    scale_x = fs_read.getNode("scale_x").real()
    scale_y = fs_read.getNode("scale_y").real()
    minx = fs_read.getNode("min_x").real()
    miny = fs_read.getNode("min_y").real()
    maxx = fs_read.getNode("max_x").real()
    maxy = fs_read.getNode("max_y").real()

    logger.info('load mapping setting - complete')
    if (is_time_log): 
        logger.info('load mapping setting took %f seconds.',time.time() - start)

    #evaluation
    start = time.time()

    test_pt_undistorted_matrix = (test_pt_matrix[:, 1:3]).copy()
    if IsNeedDistortionCorrection == True:
        test_pt_undistorted_matrix = cv.undistortPoints((test_pt_matrix[:, 1:3]).reshape(-1,1,2).astype(np.float64), cam_mat, dist_coef)
        for i in range(test_pt_undistorted_matrix.shape[0]):
            test_pt_undistorted_matrix[i][0][0] = test_pt_undistorted_matrix[i][0][0]*new_cam_mat[0, 0] + new_cam_mat[0, 2] - x
            test_pt_undistorted_matrix[i][0][1] = test_pt_undistorted_matrix[i][0][1]*new_cam_mat[1, 1] + new_cam_mat[1, 2] - y           
        
    #reproject px to px (H_px_to_px)
    reproject_pts_px = cv.perspectiveTransform(test_pt_undistorted_matrix.reshape(-1,1,2).astype(np.float64), H_px_to_px)

    #reproject px to gps (H_px_to_gps)
    logger.info("evaluate visual positioning error of H_px_to_gps")
    reproject_pts_gps = cv.perspectiveTransform(test_pt_undistorted_matrix.reshape(-1,1,2).astype(np.float64), H_px_to_gps)
    for i in range(0, gps_matrix.shape[0]):
        _GEOD = pyproj.Geod(ellps='WGS84')
        _,_,d = _GEOD.inv(gps_matrix[i][2],gps_matrix[i][1],reproject_pts_gps[i][0][1],reproject_pts_gps[i][0][0]) 
        logger.info("error of index[%f] is : %f m", gps_matrix[i][0], d)

    #reproject px to gps (H_px_to_gps2)
    logger.info("evaluate visual positioning error of H_px_to_gps2")
    reproject_pts_gps2 = cv.perspectiveTransform(test_pt_undistorted_matrix.reshape(-1,1,2).astype(np.float64), H_px_to_gps2)
    for i in range(0, gps_matrix.shape[0]):
        _GEOD = pyproj.Geod(ellps='WGS84')
    
        _,_,d = _GEOD.inv(gps_matrix[i][2],gps_matrix[i][1],reproject_pts_gps2[i][0][1],reproject_pts_gps2[i][0][0]) 
        # logger.info("ground truth gps[%f] is : %.10f, %.10f", gps_matrix[i][0], gps_matrix[i][2],gps_matrix[i][1])
        # logger.info("estimated gps[%f] is : %.10f, %.10f", gps_matrix[i][0], reproject_pts_gps2[i][0][1],reproject_pts_gps2[i][0][0])
        logger.info("error of index[%f] is : %f m", gps_matrix[i][0], d)

    logger.info('evaluation - complete')
    if (is_time_log): 
        logger.info('evaluation took %f seconds.',time.time() - start)

    #save result image
    start = time.time()
    draw_img1 = img1.copy()   
    for i in range(len(reproject_pts_px)):
        c = (int(reproject_pts_px[i][0][0]/scale_x), int(reproject_pts_px[i][0][1]/scale_y))
        cv.circle(draw_img1, c, 5, (0,0,255), 5)
    cv.imwrite("ortho_reprojection_result.png", draw_img1)

    draw_img2 = img2.copy()   
    for i in range(test_pt_undistorted_matrix.shape[0]):
        if not IsNeedDistortionCorrection:
            c = (int(test_pt_matrix[i][1]), int(test_pt_matrix[i][2])) 
        else:
            c = (int(test_pt_undistorted_matrix[i][0][0]), int(test_pt_undistorted_matrix[i][0][1]))
        cv.circle(draw_img2, c, 5, (0,0,255), 5)
    cv.imwrite("perspective_test_point.png", draw_img2)

    logger.info('save image - complete')
    if (is_time_log): 
        logger.info('save image took %f seconds.',time.time() - start)
    logger.info('================ end of logging eval.py ================')

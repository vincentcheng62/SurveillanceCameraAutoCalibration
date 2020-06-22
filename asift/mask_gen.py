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

[Mask generation module]

Module to generate a binary mask from the input ortho depth map (.tif) to assist ASIFT matching
"""


from osgeo import gdal
import numpy as np
from numpy import linalg as LA
import cv2 as cv
import sys
import time, datetime

import logging
import logging.handlers
from logging.handlers import RotatingFileHandler

def ComputeRandomIndex(img):
    """
    function to random generate sample from an image

    Note:
        N/A

    Args:
        img (M x N nparray): depth image

    Returns:
        row_idx (int), col_idx (int)

    Raises:
        N/A
    
    """

    row_idx = -1
    col_idx = -1
    while(row_idx == -1 or col_idx == -1):
        row_idx = int(np.floor(np.random.uniform(0,(img.shape)[0],1)))
        col_idx = int(np.floor(np.random.uniform(0,(img.shape)[1],1)))

        #check if this pixel is effective
        if img[row_idx, col_idx] < 0:
            row_idx = -1
            col_idx = -1

    return row_idx, col_idx

def ComputePlaneCenter(pts):
    """
    function to compute the plane center

    Note:
        N/A

    Args:
        pts (3 x 3 nparray): 3 points include row index, column index and depth

    Returns:
        plane center (1 x 3 nparray): plane center 

    Raises:
        N/A
    
    """

    return np.mean(pts, 0)

def ComputePlaneNormals(pt1, pt2, pt3):
    """
    function to compute the plane center

    Note:
        N/A

    Args:
        pt1 (1 x 3 nparray): a point that include row index, column index and depth
        pt2 (1 x 3 nparray): a point that include row index, column index and depth
        pt3 (1 x 3 nparray): a point that include row index, column index and depth

    Returns:
        n (1 x 3 nparray): surface normals of plane

    Raises:
        N/A
    
    """

    v1 = np.transpose(pt1 - pt2)
    v2 = np.transpose(pt3 - pt2)
    n = np.cross(v1, v2)
    n = n/LA.norm(n)  

    return n

def ComputePointPlaneDistance(n, plane_center, pt):
    """
    function to compute the point plane distance

    Note:
        N/A

    Args:
        n (3 x 1 nparray): surface normals of the plane
        plane_center (1 x 3 nparray): plane center
        pt (1 x 3 nparray): a point that includes row index, column index and depth

    Returns:
        point plane distance (double): point plane distance 

    Raises:
        N/A
    
    """

    v = pt - plane_center
    return np.abs(np.dot(n, v))

if __name__ == '__main__':

    loop_start_time = time.time()
    start = loop_start_time
    

    #load runtime config file
    runtime_config_read = cv.FileStorage("mask_gen_runtime_config.yml", cv.FILE_STORAGE_READ) 
    mask_config_file_path = runtime_config_read.getNode("mask_gen_config_file_path").string()
    log_config_file_path = runtime_config_read.getNode("log_config_file_path").string()

    #load log config file 
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

    is_print_to_console = log_config_read.getNode("is_print_to_console").string().lower()
    if is_print_to_console == "true":
        is_print_to_console = True
    elif is_print_to_console == "false":
        is_print_to_console = False
    else:
        print("incorrect is_print_to_console, should be either true or false")
        exit(1)

    is_time_log = log_config_read.getNode("is_time_log").string().lower()
    if is_time_log == "true":
        is_time_log = True
    elif is_time_log == "false":
        is_time_log = False
    else:
        print("incorrect is_time_log, should be either true or false")
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

    logger.info('================ start of logging mask_gen.py ================')
    logger.info('load logging config - complete')
 
    # load mask config parameters
    try:
        fs_read = cv.FileStorage(mask_config_file_path, cv.FILE_STORAGE_READ)
    except:
        logger.error('fail to load mask config')
        exit(1)

    depth_map_path = fs_read.getNode("depth_map_path").string()
    scale_x = fs_read.getNode("scale_x").real()
    scale_y = fs_read.getNode("scale_y").real()
    num_of_plane_candidate = int(fs_read.getNode("number_of_plane_candidate").real())
    pt_plane_dist_threshold = fs_read.getNode("pt_plane_dist_thres_in_m").real()
    min_scores_ratio = fs_read.getNode("min_scores_ratio").real()

    if (scale_x <= 0 or scale_x > 1):
        logger.error("wrong scale_x setting, should be within (0 - 1)")
        exit(1)

    if (scale_y <= 0 or scale_y > 1):
        logger.error("wrong scale_y setting, should be within (0 - 1)")
        exit(1)

    if (num_of_plane_candidate <= 0 or num_of_plane_candidate > 200):
        logger.error("exceeds the range of plane candidate number, should be within (0 - 200)")
        exit(1)

    if (pt_plane_dist_threshold <= 0 or pt_plane_dist_threshold > 0.3):
        logger.error("exceeds the range of point plane distance threshold, should be within (0 - 0.3)")
        exit(1)

    logger.info('load mask gen config - complete')

    # load dem
    dem = gdal.Open(depth_map_path)             
    band = dem.GetRasterBand(1) 
    d_img = band.ReadAsArray()

    logger.info('load DEM - complete')

    # down scale image for fast computation
    resized_d_img = cv.resize(d_img, (0, 0), None, scale_x, scale_y)

    logger.info('resized depth image - complete')

    # compute min scores to accept the best plane
    min_scores = 0
    for i in range(resized_d_img.shape[0]):
        for j in range(resized_d_img.shape[1]):
            if (resized_d_img[i][j] >= 0):
                min_scores = min_scores + 1

    min_scores = min_scores*min_scores_ratio

    logger.info('compute minimum scores of best plane - complete')

    if (is_time_log): 
        logger.info('load config and required inputs took %f seconds.',time.time() - start)

    logger.info("[========SUMMARY OF MASK GEN CONFIG========]")
    logger.info("depth_map_path: %s", depth_map_path)
    logger.info("scale_x: %f", scale_x)
    logger.info("scale_y: %f", scale_y)
    logger.info("num_of_plane_candidate: %f", num_of_plane_candidate)
    logger.info("pt_plane_dist_threshold: %f", pt_plane_dist_threshold)
    logger.info("min_scores_ratio: %f", min_scores_ratio)
    logger.info("min_scores: %f", min_scores)
    logger.info("[========SUMMARY OF MASK GEN CONFIG========]")

    # ransac computation
    start = time.time()
    logger.info('start searching the best plane, this process with take probably around 5 minutes')
    r_idx = np.zeros((3, 1), int)
    c_idx = np.zeros((3, 1), int)
    pts = np.zeros((3, 3), float)

    scores = np.zeros(num_of_plane_candidate, int)

    max_scores = -1
    max_idx = 0
    plane_map = np.ones(((resized_d_img.shape)[0], (resized_d_img.shape)[1], num_of_plane_candidate), float)*-1

    for candidate in range(num_of_plane_candidate):
        for i in range(len(r_idx)):
            r_idx[i], c_idx[i] = ComputeRandomIndex(resized_d_img)
            pts[i][0] = c_idx[i]
            pts[i][1] = r_idx[i]
            pts[i][2] = resized_d_img[r_idx[i], c_idx[i]]

        n = ComputePlaneNormals(pts[0], pts[1], pts[2])

        plane_center = ComputePlaneCenter(pts)

        pt = np.zeros((1,3), float)
    
        # compute point plane distance pixel by pixel (the most time - consuming session, any idea to improve?)
        for i in range((resized_d_img.shape)[0]):
            for j in range((resized_d_img.shape)[1]):
                if (resized_d_img[i][j] < 0):
                    continue

                pt[0, 0] = j
                pt[0, 1] = i
                pt[0, 2] = resized_d_img[i][j]

                pt_plane_dist = ComputePointPlaneDistance(n, plane_center, pt[0])
                
                if (pt_plane_dist <= pt_plane_dist_threshold):
                    plane_map[i][j][candidate] = 255
                    scores[candidate] = scores[candidate] + 1

        # get the best candidate
        if (max_scores < scores[candidate]):
            max_idx = candidate
            max_scores = scores[candidate]                

    #check if the extracted plane can be used as mask
    if (max_scores < min_scores):
        logger.error("this mask is not usable as the best plane does not occupy 50% of effective pixels. auto camera calibration is not recommended")
        exit(1)

    logger.info('search best plane - complete')
    if (is_time_log): 
        logger.info('search best plane took %f seconds.',time.time() - start)

    # generate mask
    start = time.time()
    result = np.zeros(((resized_d_img.shape)[0], (resized_d_img.shape)[1]), np.uint8)

    for i in range((resized_d_img.shape)[0]):
        for j in range((resized_d_img.shape)[1]):
            if (plane_map[i][j][max_idx] == 255):
                result[i][j] = 255

    result = cv.resize(result, ((d_img.shape)[1], (d_img.shape)[0]), None)
    cv.imwrite("mask.png", result)

    logger.info('mask generation - complete')
    if (is_time_log): 
        loop_end_time = time.time()
        logger.info('mask generation took %f seconds.',loop_end_time - start)
        logger.info('one loop took %f seconds.',loop_end_time - loop_start_time)

    logger.info('================ end of logging mask_gen.py ================')
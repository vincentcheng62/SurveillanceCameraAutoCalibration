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

FINAL_LINE_COLOR = (255, 255, 255)
DRAW_LINE_COLOR = (255, 0, 0)
DRAWING_THICKNESS = 4

CANVAS_SIZE = (1600.0, 900.0)

# ============================================================================

class PolygonDrawer(object):
    def __init__(self, image, logger):
        self.window_name = "Generate artificial mask on orthomosaic image"

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.logger = logger

        self.row_ratio = CANVAS_SIZE[1]/image.shape[0]
        self.col_ratio = CANVAS_SIZE[0]/image.shape[1]
        self.image = cv2.resize(image, (0,0), fx=self.col_ratio, fy=self.row_ratio) 

        # write down the instruction at top left corner
        cv2.putText(self.image, "Mouse left click to draw polgyon, double press ESC to finish drawing", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            self.logger.info('Adding point #%d with position(%d,%d)', len(self.points), x, y)
            self.points.append((x, y))
        # elif event == cv2.EVENT_RBUTTONDOWN:
        #     # Right click means we're done
        #     self.logger.info('Completing polygon with %d points.', len(self.points))
        #     self.done = True


    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window

            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(self.image, np.array([self.points]), False, DRAW_LINE_COLOR, DRAWING_THICKNESS)
                cv2.circle(self.image, (self.points[-1]), 10, DRAW_LINE_COLOR, DRAWING_THICKNESS)
                # And  also show what the current segment would look like
                #cv2.line(self.image, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, self.image)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.logger.info('Completing polygon with %d points.', len(self.points))
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        maskcanvas = np.zeros(self.image.shape, np.uint8)
        # of a filled polygon
        if (len(self.points) > 0):
            cv2.fillPoly(maskcanvas, np.array([self.points]), FINAL_LINE_COLOR)
            cv2.fillPoly(self.image, np.array([self.points]), FINAL_LINE_COLOR)
        # And show it
        cv2.imshow(self.window_name, self.image)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        resized_mask = cv2.resize(maskcanvas, (0,0), fx=(1/self.col_ratio), fy=(1/self.row_ratio))
        return cv.cvtColor(resized_mask, cv.COLOR_BGR2GRAY)

def merge_ortho_perspective(img1, img2):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)

    return vis

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
        output_dir = "./Mask_Drawing_"+str(st) +"/"
        os.mkdir(output_dir)

    is_print_to_console = log_config_read.getNode("is_print_to_console").string().lower()
    if is_print_to_console == "true":
        is_print_to_console = True
    elif is_print_to_console == "false":
        is_print_to_console = False
    else:
        print("incorrect is_print_to_console")
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

    logger.info('================ start of logging gen_artificial_mask.py ================')
    logger.info('load logging config - complete')

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

    logger.info('load match config - complete')

    #load img and geotiff
    start = time.time()
    orthophoto_img_path = match_config_read.getNode("orthophoto_img_path").string() 
    perspective_img_path = match_config_read.getNode("perspective_img_path").string()
    geotiff_path = match_config_read.getNode("geotiff_path").string()

    if orthophoto_img_path == "" or perspective_img_path == "" or geotiff_path == "":
        logger.error("empty image/geotiff path")
        exit(1)

    img1 = cv.imread(orthophoto_img_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv.imread(perspective_img_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        logger.error('Failed to load orthophoto: %s', orthophoto_img_path)
        exit(1)

    if img2 is None:
        logger.error('Failed to load perspective image: %s', perspective_img_path)
        exit(1)

    logger.info('Size of ortho : %s %s', str(img1.shape[1]), str(img1.shape[0]))
    logger.info('Size of perspective : %s %s', str(img2.shape[1]), str(img2.shape[0]))   

    logger.info("[========SUMMARY OF MATCH CONFIG========]")
    logger.info("orthophoto_img_path: %s", orthophoto_img_path)
    logger.info("perspective_img_path: %s", perspective_img_path)
    logger.info("[========END SUMMARY OF MATCH CONFIG========]")

    logger.info('Start merging ortho and perspective into one view...')
    merged = merge_ortho_perspective(img1, img2)

    logger.info('Start drawing!!!')
    pd = PolygonDrawer(merged, logger)
    outputmask = pd.run()
    logger.info('Mask polygon vertex list = %s', pd.points)

    f = open(os.path.join(output_dir,"mask_polygon_vertex_list.txt"), "a")
    f.write(str(pd.points))
    f.close()

    crop_mask_ortho = outputmask[0:img1.shape[0], 0:img1.shape[1]]
    crop_mask_persp = outputmask[0:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]]

    if np.sum(crop_mask_ortho) > 0:
        cv2.imwrite(os.path.join(output_dir,"outputmask_ortho.png"), crop_mask_ortho)
        cv2.imwrite(os.path.join(output_dir, "raw_ortho_and_mask.png"), cv2.bitwise_and(img1, crop_mask_ortho))
        logger.info('The output mask for ortho image is generated')

    if np.sum(crop_mask_persp) > 0:
        cv2.imwrite(os.path.join(output_dir,"outputmask_persp.png"), crop_mask_persp)
        cv2.imwrite(os.path.join(output_dir, "raw_persp_and_mask.png"), cv2.bitwise_and(img2, crop_mask_persp))
        logger.info('The output mask for perspective image is generated')

    logger.info("Output mask is saved in %s", output_dir)



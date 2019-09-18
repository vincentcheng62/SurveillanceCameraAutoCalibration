
"""
Module to calibrate the intrinsic matrix and distortion coefficient of the camera
Originate from opencv calibrate.py

"""

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local modules
#from common import splitfn


import imutils
import os
import sys
import getopt
from glob import glob


def splitfn(file_path):
    """
    function to split a long path to different parts
    """
    file_path_parts = file_path.split(sep=os.sep)
    _path = os.path.join(*file_path_parts[:-1])
    file_name = file_path_parts[-1]
    file_name_parts = file_name.split(sep='.')
    return _path, file_name_parts[0], file_name_parts[1]

if __name__ == '__main__':


    # read parameters from intrinsic_calibration_input.yml
    print("Reading input from intrinsic_calibration_input.yml...")
    fs_read = cv.FileStorage("intrinsic_calibration_input.yml", cv.FILE_STORAGE_READ)

    square_size_in_m = fs_read.getNode("square_size_in_m").real()
    print("square_size_in_m: ", square_size_in_m)
    IsRotated = eval(fs_read.getNode("IsRotated").string())
    print("IsRotated: ", IsRotated)    
    pattern_size_x = int(fs_read.getNode("pattern_size_x").real())
    print("pattern_size_x: ", pattern_size_x)
    pattern_size_y = int(fs_read.getNode("pattern_size_y").real())
    print("pattern_size_y: ", pattern_size_y)   
    debugdir = fs_read.getNode("debugdir").string()
    print("debugdir: ", debugdir)
    img_mask = fs_read.getNode("img_mask").string()
    print("img_mask: ", img_mask)
    number_of_thread = int(fs_read.getNode("number_of_thread").real())
    print("number_of_thread: ", number_of_thread)   
    fs_read.release()


    img_names = glob(img_mask)
    debug_dir = debugdir
    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)

    pattern_size = (pattern_size_x, pattern_size_y)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size_in_m

    obj_points = []
    img_points = []
    w=100
    h=100

    if IsRotated:
        w, h = cv.imread(img_names[0], 0).shape[:2]  # TODO: use imquery call to retrieve results
    else:
        h, w = cv.imread(img_names[0], 0).shape[:2]  # TODO: use imquery call to retrieve results

    print("")
    print("Start intrinsic calibration...")

    def processImage(fn):
        print('processing %s... ' % fn)
        img = cv.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            return None

        #assert h == img.shape[1] and w == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        if IsRotated:
            img = imutils.rotate_bound(img, -90)

        found, corners = cv.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = splitfn(fn)
            outfile = os.path.join(debug_dir, name + '_chess.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('chessboard not found')
            return None

        print('           %s... OK' % fn)
        return (corners.reshape(-1, 2), pattern_points)

    if number_of_thread <= 1:
        chessboards = [processImage(fn) for fn in img_names]
    else:
        print("Run with %d threads..." % number_of_thread)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(number_of_thread)
        chessboards = pool.map(processImage, img_names)

    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None, flags=cv.CALIB_RATIONAL_MODEL)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    fs_write = cv.FileStorage('intrinsic_and_distortion_coeff.yml', cv.FILE_STORAGE_WRITE)
    fs_write.write("RMS", rms)   
    fs_write.write("intrinsic", camera_matrix)   
    fs_write.write("distortion_coeff", dist_coefs)   
    fs_write.release()

    # undistort the image with the calibration
    print('')
    for fn in img_names if debug_dir else []:
        path, name, ext = splitfn(fn)
        img_found = os.path.join(debug_dir, name + '_chess.png')
        outfile = os.path.join(debug_dir, name + '_undistorted.png')

        img = cv.imread(img_found)
        if img is None:
            continue

        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
        #dst = cv.undistort(img, camera_matrix, dist_coefs, None, None)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        print('Undistorted image written to: %s' % outfile)
        cv.imwrite(outfile, dst)

    cv.destroyAllWindows()
    print("End of intrinsic calibration...")

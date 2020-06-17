import cv2
import math
import pyproj
import numpy as np

IsNeedDistortionCorrection=False

print("Input from src_dst_points.yml...")
print("[INPUT]")
fs_read = cv2.FileStorage("src_dst_points.yml", cv2.FILE_STORAGE_READ)
gps_matrix = fs_read.getNode("gps_matrix").mat()
print("gps_matrix: ", gps_matrix)
cross_subpixel_matrix = fs_read.getNode("cross_subpixel_matrix").mat()
print("cross_subpixel_matrix: ", cross_subpixel_matrix)
fs_read.release()

fs_read = cv2.FileStorage("intrinsic_and_distortion_coeff.yml", cv2.FILE_STORAGE_READ)
camera_matrix_manual = fs_read.getNode("intrinsic").mat()
print("intrinsic: ", camera_matrix_manual)
dist_coefs_manual = fs_read.getNode("distortion_coeff").mat()
print("distortion_coeff: ", dist_coefs_manual)
fs_read.release()  

cross_subpixel_undistorted_matrix = (cross_subpixel_matrix[:, 1:3]).copy()
if IsNeedDistortionCorrection:
    cross_subpixel_undistorted_matrix = cv2.undistortPoints((cross_subpixel_matrix[:, 1:3]).reshape(-1,1,2).astype(np.float64), camera_matrix_manual, dist_coefs_manual)
    print("cross_subpixel_undistorted_matrix: ", cross_subpixel_undistorted_matrix)

ransacReprojThreshold = 5.0
# Maximum allowed reprojection error to treat a point pair as an inlier
H, mask = cv2.findHomography(cross_subpixel_undistorted_matrix, gps_matrix[:, 1:3], cv2.RANSAC, ransacReprojThreshold)
print(" ")
print("[OUTPUT]")
print("homography matrix: ", H)
print("Calibration of homography matrix is done, the result is stored in homography.yml")

fs_write = cv2.FileStorage('homography.yml', cv2.FILE_STORAGE_WRITE)
fs_write.write("homography_matrix", H)
fs_write.release()




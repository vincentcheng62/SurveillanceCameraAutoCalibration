import cv2
import math
import pyproj
import numpy as np

IsNeedDistortionCorrection=False

print("Input from src_dst_test_points.yml...")
print("[INPUT]")
fs_read = cv2.FileStorage("src_dst_test_points.yml", cv2.FILE_STORAGE_READ)
gps_matrix = fs_read.getNode("gps_matrix").mat()
print("gps_matrix: ", gps_matrix)
cross_subpixel_matrix = fs_read.getNode("cross_subpixel_matrix").mat()
print("cross_subpixel_matrix: ", cross_subpixel_matrix)

fs_read = cv2.FileStorage("homography.yml", cv2.FILE_STORAGE_READ)
homography_matrix = fs_read.getNode("homography_matrix").mat()
fs_read.release()  

cross_subpixel_undistorted_matrix = (cross_subpixel_matrix[:, 1:3]).copy()
if IsNeedDistortionCorrection:
    cross_subpixel_undistorted_matrix = cv2.undistortPoints((cross_subpixel_matrix[:, 1:3]).reshape(-1,1,2).astype(np.float64), camera_matrix_manual, dist_coefs_manual)
    print("cross_subpixel_undistorted_matrix: ", cross_subpixel_undistorted_matrix)

gps_coord = cv2.perspectiveTransform(cross_subpixel_undistorted_matrix.reshape(-1,1,2).astype(np.float64), homography_matrix)
print(" ")
print("[OUTPUT]")

for i in range(0, gps_matrix.shape[0]):
    _GEOD = pyproj.Geod(ellps='WGS84')
    _,_,d = _GEOD.inv(gps_matrix[i][2],gps_matrix[i][1],gps_coord[i][0][1],gps_coord[i][0][0]) 
    print("pyproj error of index[" , gps_matrix[i][0], "] is :", d, "m")




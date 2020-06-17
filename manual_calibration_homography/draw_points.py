import cv2
import math
import pyproj
import numpy as np

IsNeedDistortionCorrection=False

frame = cv2.imread("mark.png")

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

## start the drawing

for pt in cross_subpixel_matrix:
    #print(pt)
    cv2.circle(frame,(int(pt[1]), int(pt[2])), 10, (0, 0, 255), -1)
    cv2.putText(frame, str(int(pt[0])), (int(pt[1]) ,int(pt[2])), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 1, cv2.LINE_AA)
cv2.imwrite("draw_points_result.jpg", frame)
print("Finish drawing! the result is saved to draw_points_result.jpg")




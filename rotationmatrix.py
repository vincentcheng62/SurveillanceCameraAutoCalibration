import cv2
import math
import pyproj
import numpy as np

fs_read = cv2.FileStorage("rotationmatrix.yml", cv2.FILE_STORAGE_READ)
matrix = fs_read.getNode("matrix").mat()
print("rotation_matrix: ", matrix)
fs_read.release()

dst, Jacobian = cv2.Rodrigues(matrix)
print("rotation: ", math.degrees(dst[0]), math.degrees(dst[1]), math.degrees(dst[2]))
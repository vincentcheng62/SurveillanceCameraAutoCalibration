import cv2
import math
import pyproj
import numpy as np

fs_read = cv2.FileStorage("gps_corner_calib.yml", cv2.FILE_STORAGE_READ)
antenna_height = fs_read.getNode("antenna_height").real()
print("antenna_height: ", antenna_height)
square_size = fs_read.getNode("square_size").real()
print("square_size: ", square_size)
cb_x_axis_to_north = fs_read.getNode("cb_x_axis_to_north").real()
print("cb_x_axis_to_north (in degree)(anti-clockwise): ", cb_x_axis_to_north)
gps_matrix = fs_read.getNode("gps_matrix").mat()
print("gps_matrix: ", gps_matrix)
world_coord_matrix = fs_read.getNode("world_coord_matrix").mat()
print("world_coord_matrix: ", world_coord_matrix)
fs_read.release()

IsUsingCampusNorthCalibrationMethod=True

def gps_to_ecef_pyproj(lat, lon, alt):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)

    return x, y, z

# Transform gps coord to ecef coord
ecef_matrix = gps_matrix.copy()
for zz in range(0, gps_matrix.shape[0]):
    x, y, z = gps_to_ecef_pyproj(gps_matrix[zz][0], gps_matrix[zz][1], gps_matrix[zz][2]-antenna_height)
    ecef_matrix[zz][0]=x
    ecef_matrix[zz][1]=y
    ecef_matrix[zz][2]=z

print("ecef_matrix: ", ecef_matrix)

#Solve 3d affine transformation from chessboard coord to ECEF coord using estimateAffine3D
retval, M, inliers = cv2.estimateAffine3D(world_coord_matrix, ecef_matrix)
if retval:
    print("affine transform from CB to ECEF: ", M)
    dst, Jacobian = cv2.Rodrigues(M[:,0:3])
    print("rotation: ", np.transpose(dst))
    print("inliers: ", cv2.transpose(inliers))

    # Calculate the Direction Cosine Matrix

    lat=math.radians(gps_matrix[0][0])
    lon=math.radians(gps_matrix[0][1])
    R_From_NED_TO_ECEF = np.zeros((3, 3), np.float32)
    R_From_NED_TO_ECEF[0, 0] = -1*math.sin(lat)*math.cos(lon)
    R_From_NED_TO_ECEF[1, 0] = -1*math.sin(lat)*math.sin(lon)
    R_From_NED_TO_ECEF[2, 0] = math.cos(lat)
    R_From_NED_TO_ECEF[0, 1] = -1*math.sin(lon)
    R_From_NED_TO_ECEF[1, 1] = math.cos(lon)
    R_From_NED_TO_ECEF[2, 1] = 0
    R_From_NED_TO_ECEF[0, 2] = -1*math.cos(lat)*math.cos(lon)
    R_From_NED_TO_ECEF[1, 2] = -1*math.cos(lat)*math.sin(lon)
    R_From_NED_TO_ECEF[2, 2] = -1*math.sin(lat)

    print("R_From_NED_TO_ECEF: ", R_From_NED_TO_ECEF)
    dst2, Jacobian2 = cv2.Rodrigues(R_From_NED_TO_ECEF)
    print("R_From_NED_TO_ECEF 3 axis rot: ", np.transpose(dst2))

    rvec_From_CB_TO_NED = np.zeros((3, 1), np.float32)
    rvec_From_CB_TO_NED[0][0]=0
    rvec_From_CB_TO_NED[1][0]=0
    rvec_From_CB_TO_NED[2][0]=math.radians(cb_x_axis_to_north)

    print("rvec_From_CB_TO_NED: ", np.transpose(rvec_From_CB_TO_NED))
    dst3, Jacobian3 = cv2.Rodrigues(rvec_From_CB_TO_NED)
    print("rvec_From_CB_TO_NED rotation matrix: ",dst3)

    dst4 = np.matmul(R_From_NED_TO_ECEF, dst3)
    print("rotation matrix from CB TO ECEF by campus: ", dst4)
    dst5, Jacobian5 = cv2.Rodrigues(dst4)
    print("campus rotation in rvec: ", np.transpose(dst5))

    if IsUsingCampusNorthCalibrationMethod:
        M[:,0:3]=dst4
        print("Using campus north calibrated rotation matrix!!! Update M!")

        # self test
        TEST_M = np.ones((4, 4), np.float32)
        TEST_M[0:3,:]=np.transpose(world_coord_matrix)
        print("TEST_M: ", TEST_M)
        TEST_RESULT=np.transpose(np.matmul(M, TEST_M))
        print("TEST_RESULT: ", TEST_RESULT)  
        DIFF=ecef_matrix-TEST_RESULT
        print("ecef DIFF: ", DIFF)            

    fs_write = cv2.FileStorage('cb_to_ecef.yml', cv2.FILE_STORAGE_WRITE)
    fs_write.write("transform", M)
    fs_write.write("inliers", inliers)
    fs_write.release()





"""
Module to calibrate the transformation from chessboard coordinate to ECEF coordinate.
"""

import cv2
import math
import pyproj
import numpy as np


def gps_to_ecef_pyproj(lat, lon, alt):
    """
    function to convert from gps polar coordinate to ECEF euclidean coordinates.
    """
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)

    return x, y, z

if __name__ == '__main__':

    # read parameters from gps_calibration_input.yml
    print("Reading input from gps_calibration_input.yml...")
    fs_read = cv2.FileStorage("gps_calibration_input.yml", cv2.FILE_STORAGE_READ)
    
    antenna_height_in_m = fs_read.getNode("antenna_height_in_m").real()
    print("antenna_height_in_m: ", antenna_height_in_m)
    IsUsingCampusNorthCalibrationMethod = eval(fs_read.getNode("IsUsingCampusNorthCalibrationMethod").string())
    print("IsUsingCampusNorthCalibrationMethod: ", IsUsingCampusNorthCalibrationMethod)    
    cb_x_axis_to_north_in_deg = fs_read.getNode("cb_x_axis_to_north_in_deg").real()
    print("cb_x_axis_to_north_in_deg(anti-clockwise): ", cb_x_axis_to_north_in_deg)
    gps_matrix = fs_read.getNode("gps_matrix").mat()
    print("gps_matrix: ", gps_matrix)
    world_coord_matrix = fs_read.getNode("world_coord_matrix").mat()
    print("world_coord_matrix: ", world_coord_matrix)
    fs_read.release()


    # Transform input gps coord to ecef coord
    ecef_matrix = gps_matrix.copy()
    for zz in range(0, gps_matrix.shape[0]):
        x, y, z = gps_to_ecef_pyproj(gps_matrix[zz][0], gps_matrix[zz][1], gps_matrix[zz][2]-antenna_height_in_m)
        ecef_matrix[zz][0]=x
        ecef_matrix[zz][1]=y
        ecef_matrix[zz][2]=z

    print("ecef_matrix: ", ecef_matrix)


    fs_write = cv2.FileStorage('gps_calibration_result.yml', cv2.FILE_STORAGE_WRITE)
    if IsUsingCampusNorthCalibrationMethod:

        print("Method: Using campus north calibration method")

        # Flow: Chessboard->NED->ECEF->GPS
    
        # Calculate the Direction Cosine Matrix
        # according to wikipedia https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates        

        lat=math.radians(gps_matrix[0][0])
        lon=math.radians(gps_matrix[0][1])

        Rot_From_NED_TO_ECEF = np.zeros((3, 3), np.float32)
        Rot_From_NED_TO_ECEF[0, 0] = -1*math.sin(lat)*math.cos(lon)
        Rot_From_NED_TO_ECEF[1, 0] = -1*math.sin(lat)*math.sin(lon)
        Rot_From_NED_TO_ECEF[2, 0] = math.cos(lat)
        Rot_From_NED_TO_ECEF[0, 1] = -1*math.sin(lon)
        Rot_From_NED_TO_ECEF[1, 1] = math.cos(lon)
        Rot_From_NED_TO_ECEF[2, 1] = 0
        Rot_From_NED_TO_ECEF[0, 2] = -1*math.cos(lat)*math.cos(lon)
        Rot_From_NED_TO_ECEF[1, 2] = -1*math.cos(lat)*math.sin(lon)
        Rot_From_NED_TO_ECEF[2, 2] = -1*math.sin(lat)

        # NED= North-East-Down coordinate system
        print("Rot_From_NED_TO_ECEF: ", Rot_From_NED_TO_ECEF)
        rvec_From_NED_TO_ECEF, _ = cv2.Rodrigues(Rot_From_NED_TO_ECEF)
        print("rvec_From_NED_TO_ECEF: ", np.transpose(rvec_From_NED_TO_ECEF))

        # transformation between chessboard coordinate to NED coordinate, only differs by a rotation in Z-axis with cb_x_axis_to_north
        rvec_From_CB_TO_NED = np.zeros((3, 1), np.float32)
        rvec_From_CB_TO_NED[0][0]=0
        rvec_From_CB_TO_NED[1][0]=0
        rvec_From_CB_TO_NED[2][0]=math.radians(cb_x_axis_to_north_in_deg)

        print("rvec_From_CB_TO_NED: ", np.transpose(rvec_From_CB_TO_NED))
        Rot_from_CB_TO_NED, _ = cv2.Rodrigues(rvec_From_CB_TO_NED)
        print("Rot_from_CB_TO_NED: ",Rot_from_CB_TO_NED)

        Rot_from_CB_TO_ECEF = np.matmul(Rot_From_NED_TO_ECEF, Rot_from_CB_TO_NED)
        print("Rot_from_CB_TO_ECEF: ", Rot_from_CB_TO_ECEF)
        rvec_From_CB_TO_ECEF, _ = cv2.Rodrigues(Rot_from_CB_TO_ECEF)
        print("rvec_From_CB_TO_ECEF: ", np.transpose(rvec_From_CB_TO_ECEF))

        cb_to_ecef = np.zeros((3, 4), np.float32)
        cb_to_ecef[:,0:3]=Rot_from_CB_TO_ECEF

        # just pick the first row in ecef_matrix to be the rtk reference
        cb_to_ecef[:,3]=np.transpose(ecef_matrix[0])

        print("transform from CB to ECEF: ", cb_to_ecef)
        fs_write.write("cb_to_ecef transform", cb_to_ecef)   
        
    else:

        print("Method: Using points to points estimate affine transformation method")       

        #Solve 3d affine transformation from chessboard coord to ECEF coord using estimateAffine3D
        retval, cb_to_ecef, inliers = cv2.estimateAffine3D(world_coord_matrix, ecef_matrix)
        if retval:
            print("affine transform from CB to ECEF: ", cb_to_ecef)
            fs_write.write("cb_to_ecef transform", cb_to_ecef)            
            rvec, _ = cv2.Rodrigues(cb_to_ecef[:,0:3])
            print("rotation in rvec: ", np.transpose(rvec))
            print("inliers: ", cv2.transpose(inliers))
            fs_write.write("inliers", inliers)
        else:
            print("Fail to perform cv2.estimateAffine3D() from world coord matrix to ecef matrix, please check the input!")



    fs_write.release()
    print("End of gps calibration...")




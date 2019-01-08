import cv2
import math
import pyproj

fs_read = cv2.FileStorage("gps_corner_calib.yml", cv2.FILE_STORAGE_READ)
antenna_height = fs_read.getNode("antenna_height").real()
print("antenna_height: ", antenna_height)
square_size = fs_read.getNode("square_size").real()
print("square_size: ", square_size)
gps_matrix = fs_read.getNode("gps_matrix").mat()
print("gps_matrix: ", gps_matrix)
world_coord_matrix = fs_read.getNode("world_coord_matrix").mat()
print("world_coord_matrix: ", world_coord_matrix)
fs_read.release()


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
    print("affine transform: ", M)
    dst, Jacobian = cv2.Rodrigues(M[:,0:3])
    print("rotation: ", dst)
    print("inliers: ", cv2.transpose(inliers))

    fs_write = cv2.FileStorage('cb_to_ecef.yml', cv2.FILE_STORAGE_WRITE)
    fs_write.write("transform", M)
    fs_write.write("inliers", inliers)
    fs_write.release()




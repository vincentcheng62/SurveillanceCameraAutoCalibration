import cv2
import math
import pyproj

# in windows, conda install -c conda-forge pyproj

fs_read = cv2.FileStorage("twogps.yml", cv2.FILE_STORAGE_READ)
lat1 = fs_read.getNode("lat1").real()
lon1 = fs_read.getNode("lon1").real()
print("lat1, lon1: ", lat1, ", ", lon1)
lat2 = fs_read.getNode("lat2").real()
lon2 = fs_read.getNode("lon2").real()
print("lat2, lon2: ", lat2, ", ", lon2)
fs_read.release()

_GEOD = pyproj.Geod(ellps='WGS84')
a,a2,d = _GEOD.inv(lon1,lat1,lon2,lat2) 
print("pyproj   dist:", d, "m")









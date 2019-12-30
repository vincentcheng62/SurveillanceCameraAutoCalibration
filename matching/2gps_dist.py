import cv2
import math
import pyproj

fs_read = cv2.FileStorage("twogps.yml", cv2.FILE_STORAGE_READ)
lat1 = fs_read.getNode("lat1").real()
lon1 = fs_read.getNode("lon1").real()
print("lat1, lon1: ", lat1, ", ", lon1)
lat2 = fs_read.getNode("lat2").real()
lon2 = fs_read.getNode("lon2").real()
print("lat2, lon2: ", lat2, ", ", lon2)
fs_read.release()


earthRadiusKm = 6371

_GEOD = pyproj.Geod(ellps='WGS84')
a,a2,d = _GEOD.inv(lon1,lat1,lon2,lat2) 
print("pyproj   dist:", d, "m")

dLat = math.radians(lat2-lat1)
dLon = math.radians(lon2-lon1)

lat1 = math.radians(lat1)
lat2 = math.radians(lat2)

a = math.sin(dLat/2) * math.sin(dLat/2) + math.sin(dLon/2) * math.sin(dLon/2) * math.cos(lat1) * math.cos(lat2); 
c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)); 
dist = earthRadiusKm * c *1000.0
print("opensrc dist: ", dist, "m")








from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image
import time, datetime
import msgpack
import json
import numpy
import base64
import urllib.parse

import numpy as np
import cv2 as cv
import cv2
from matplotlib import pyplot as plt
import operator
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linspace
import argparse
from imutils.object_detection import non_max_suppression
from numpy.linalg import inv
from math import log10, floor
import pyproj

#kafka
from kafka import KafkaProducer
from kafka.errors import KafkaError
import msgpack
import json
import pymap3d as pm
#producer = KafkaProducer(bootstrap_servers=['172.18.6.120:31090'], value_serializer=lambda m: json.dumps(m).encode('utf-8'))

# load weights and set defaults
config_path='config/yolov3-spp.cfg'
weights_path='config/yolov3-spp.weights'
#config_path='config/yolov3.cfg'
#weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=608 #416  #608
conf_thres=0.2 # higher means less object detect
nms_thres=0.4
frame_rate=29
Seconds_to_process=10
total_frame=frame_rate*Seconds_to_process

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

def degreesToRadians(degrees):
  return degrees * math.pi / 180

earthRadiusKm = 6371


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv

def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    M[1,1] = 0
    M[1,2] = 1
    M[2,1] = -1
    M[2,2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))

def round_to_1(x, sig=2):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)

def GetWindowWithAxis(Size_of_w, physical_size):
    int_size = int(Size_of_w*0.5)
    center = (int_size, int_size)
    px_of_meter = int(Size_of_w/physical_size)
    img = np.zeros((Size_of_w, Size_of_w, 3), np.uint8)
    img = cv.line(img, center, (center[0], center[1]+px_of_meter), (0,0,255), 3)
    img = cv.line(img, center, (center[0]-px_of_meter, center[1]), (0,255,0), 3)
    return img

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

#line in form of y=ax+c , so a tuple (a, c)
#return (IsHavingIntersection, inter_x, inter_y)
def find_two_line_intersection(line1, line2):
    if line1[0] == line2[0]:
        #print("The line are parallel!")
        return (False, -1, -1)
    else:
        a = line1[0]
        c = line1[1]
        b = line2[0]
        d = line2[1]
        return (True, (d-c)/(a-b), (a*d-b*c)/(a-b))

def GetBottomPtScale(head_pt, bottom_pt, rot, ref_tvec, method=0):
    s1=1
    s2=1
    r13=rot[0][2]
    r23=rot[1][2]
    r33=rot[2][2]
    t1=ref_tvec[0][0]
    t2=ref_tvec[1][0]
    t3=ref_tvec[2][0]
    u1=head_pt[0]
    v1=head_pt[1]
    u2=bottom_pt[0]
    v2=bottom_pt[1]
    h=-1.7

    if method==0:
        # Assume the world coord of the bottom pt (X, Y, Z=0), then there is unique solution for s2
        Tz = r13*t1+r23*t2+r33*t3
        dz = r13*u2+r23*v2+r33*1.0
        s2 = Tz/dz

        #Also the head pt projection along cam optic center to estimate human height
        Tzh = r13*t1+r23*t2+r33*t3
        dzh = r13*u1+r23*v1+r33*1.0
        s1 = Tzh/dzh        
        #print("s2g: ", s2)

    else:
        # Use head-pt and bottom-pt to approximate the (X, Y, Z) of the bottom pt in which Z != 0
        # Need to minimize f where f(s1, s2) = Norm((s1u1-s2u2-hr13, s1v1-s2v2-hr23, s1-s2-hr33))
        # That means, want to find a pair of (s1, s2) such that their world coord (X1, Y1, Z1), (X2, Y2, Z2)
        # can achieve X1 close to X2, Y1 close to Y2, Z1-Z2 close to h=1.7m normal human height
        # Taking df/ds1 and df/ds2=0 and solve for s1 and s2
        # Since u1=u2,...


        Df = np.zeros((2, 2), np.float32)
        Df[0, 0] = 2*(u1*u1+v1*v1+1)
        Df[0, 1] = -2*(u1*u1+v1*v2+1)
        Df[1, 0] = -2*(u1*u1+v1*v2+1)
        Df[1, 1] = 2*(u1*u1+v2*v2+1)
        #print("Df: ", Df)

        Zero = np.zeros((2, 1), np.float32)
        Zero[0, 0] = 2*h*(u1*r13+v1*r23+r33)
        Zero[1, 0] = -2*h*(u1*r13+v2*r23+r33)
        #print("Zero: ", Zero)

        Answer = np.matmul(inv(Df), Zero)
        #print("Answer: ", Answer)
        s1=Answer[0][0]
        s2=Answer[1][0]

        #Calculate the loss vector
        rhs = np.zeros((3, 1), np.float32)
        rhs[0, 0] = h*r13
        rhs[1, 0] = h*r23
        rhs[2, 0] = h*r33

        lhs = np.zeros((3, 1), np.float32)
        lhs[0, 0] = s1*u1-s2*u2
        lhs[1, 0] = s1*v1-s2*v2
        lhs[2, 0] = s1-s2

        #print("loss vector: ", lhs-rhs)    
        diff = np.matmul(inv(rot), lhs)
        #print("diff vector: ", cv.transpose(diff))    
        #print("s1: ", s1, ", s2: ", s2)

    return s1, s2

ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
def RecoverGPSCoord(cb_coord):
    ecef_coord = np.matmul(cb_to_ecef_transform[:,0:3], cb_coord) + cv.transpose(np.array([cb_to_ecef_transform[:, 3]]))
    #print("ecef_coord: ", cv.transpose(ecef_coord))
    #lon, lat, alt = pyproj.transform(ecef, lla, ecef_coord[0][0], ecef_coord[1][0], ecef_coord[2][0], radians=False)
    lat,lon,alt = pm.ecef2geodetic(ecef_coord[0][0], ecef_coord[1][0], ecef_coord[2][0])
    return (lat, lon, alt)

    #return (ecef_coord[0][0], ecef_coord[1][0], ecef_coord[2][0])

fs_read = cv2.FileStorage("cb_to_ecef.yml", cv2.FILE_STORAGE_READ)
cb_to_ecef_transform = fs_read.getNode("transform").mat()
print("cb_to_ecef_transform: ", cb_to_ecef_transform)
fs_read.release()  

camera_matrix_manual = np.zeros((3, 3), np.float32)
camera_matrix_manual[0, 0] = 1009.60665
#camera_matrix_manual[1, 0] = 0
#camera_matrix_manual[2, 0] = 0
#camera_matrix_manual[0, 1] = 0
camera_matrix_manual[1, 1] = 1009.32417
#camera_matrix_manual[2, 1] = 0
camera_matrix_manual[0, 2] = 651.53609
camera_matrix_manual[1, 2] = 336.868
camera_matrix_manual[2, 2] = 1

ref_rvec = np.zeros((3, 1), np.float32)     
ref_rvec[0, 0] = 0.1
ref_rvec[1, 0] = -0.1
ref_rvec[2, 0] = -0.5
rot, jaco = cv.Rodrigues(ref_rvec)

ref_tvec = np.zeros((3, 1), np.float32)     
ref_tvec[0, 0] = 5.2
ref_tvec[1, 0] = 1.6
ref_tvec[2, 0] = -3.4

human_height=1.7

#videopath = '../data/video/overpass.mp4'
#videopath = 'IMG_6308.MOV'
videopath = 'traffic2_trim.mp4'
#videopath = 'traffic_night9.mp4'
#videopath = 'traffic2_trim_1080p.mp4'
# file_to_write = 'kafka_json.txt'
# filew = open(file_to_write,'w')

#videopath = 'test_far_ppl3.mp4'

import cv2
from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

#cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Stream', (800,600))

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
#outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det17.avi"),fourcc,20.0,(vw,vh))

dist_coefs_manual = np.zeros((1, 8), np.float32)     
dist_coefs_manual[0, 0] = -6.08059316
dist_coefs_manual[0, 1] = 9.70169024
dist_coefs_manual[0, 2] = 1.60141342e-03
dist_coefs_manual[0, 3] = -6.39510521e-05 
dist_coefs_manual[0, 4] = -1.77135020  
dist_coefs_manual[0, 5] = -5.71916015  
dist_coefs_manual[0, 6] = 7.55768940  
dist_coefs_manual[0, 7] = 1.36953813  


frames = 0
counter=0
starttime = time.time()
regionpts = numpy.array([[85,109],[55,225],[582,236],[495,127]], numpy.int32)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

while(counter<total_frame):

    realstart = time.time()
    counter=counter+1
    print("counter: ", counter)

    start = time.time()
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    print("read frame took {} seconds.".format(time.time() - start))

    start = time.time()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    print("convert color and run detection took {} seconds.".format(time.time() - start))

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    #retval, buffer = cv2.imencode('.jpeg', frame, encode_param)

    object_array=[]

    if detections is not None:

        start = time.time()
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        print("tracker update took {} seconds.".format(time.time() - start))

        print("Number of objects detected: ", len(tracked_objects))

        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:

            starto = time.time()
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            bb=(x1, y1, box_w, box_h)
            #color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]
            #cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            #cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
            #cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

            #construct json
            objecta={}
            objecta['isHit']=[]
            objecta['isHit'].append(False)
            objecta['isHit'].append(True)
            objecta['type']=str(cls)
            objecta['moving_direction']=0.315
            objecta['speed_in_m']=0.56
            objecta['id']=obj_id
            objecta['confidence']=0.99
            objecta['boundingbox']={'XXX': x1, 'YYY': y1, 'ZZZ': box_w, 'VVV': box_h}

        
            #Localization
            xA = x1
            yA = y1
            xB = x1+box_w
            yB = y1+box_h
            ratio=1.0

            head_pt = [(xA+xB)*0.5*ratio, min(yA, yB)*ratio] # (u1, v1)
            bottom_pt = [(xA+xB)*0.5*ratio, max(yA, yB)*ratio] # (u2, v2)

            bp = np.zeros((1, 2), np.float32)
            bp[0][0] = bottom_pt[0]
            bp[0][1] = bottom_pt[1]
            hp = np.zeros((1, 2), np.float32)
            hp[0][0] = head_pt[0]
            hp[0][1] = head_pt[1]        
            #print("calib_corners: ", calib_corners[0])
            dst = cv.undistortPoints(bp.reshape(-1,1,2).astype(np.float32), camera_matrix_manual, dist_coefs_manual)
            bottom_pt = dst[0][0]
            dst2 = cv.undistortPoints(hp.reshape(-1,1,2).astype(np.float32), camera_matrix_manual, dist_coefs_manual)        
            head_pt = dst2[0][0]
            # according to the formula: s1=1.7*r33+s2, s2=1.7*(r23-v1*r33)/(v1-v2)
            #s2 = 1.7*(rot[1][2]-head_pt[1]*rot[2][2])/(head_pt[1]-bottom_pt[1])
            #bottom_pt_center_normalized = (bottom_pt[0]-camera_matrix_manual[0][2], bottom_pt[1]-camera_matrix_manual[1][2])
            #print("bottom_pt_center_normalized: ", bottom_pt_center_normalized)        
            #s1 = 1.7*rot[2][2]+s2

            #print("ref_tvec: ", ref_tvec[0][0], ref_tvec[1][0], ref_tvec[2][0])
            #AlgMethod=1  # 0 : assume z=0, 1: assume height=1.7
            #start = time.time()
            s1g, s2g = GetBottomPtScale(head_pt, bottom_pt, rot, ref_tvec, 0)
            #print("One object getbottomscale took {} seconds.".format(time.time() - start))
            #s1, s2 = GetBottomPtScale(head_pt, bottom_pt, rot, ref_tvec, 1)

            img_pt_h = np.array([[head_pt[0]], [head_pt[1]], [1]])
            img_pt_b = np.array([[bottom_pt[0]], [bottom_pt[1]], [1]])
            #print("zz: ", s2*img_pt-ref_tvec)
            #print("tra: ", cv.transpose(rot))
            # result1 = np.matmul(cv.transpose(rot), s1*img_pt_h-ref_tvec)
            # result2 = np.matmul(cv.transpose(rot), s2*img_pt_b-ref_tvec)
            # resulthp = np.matmul(cv.transpose(rot), s1g*img_pt_b-ref_tvec)
            resultg = np.matmul(cv.transpose(rot), s2g*img_pt_b-ref_tvec)

            # ground_dist=math.sqrt(math.pow(resultg[0]-resulthp[0],2)+math.pow(resultg[1]-resulthp[1],2))
            # cam_height=CamCenterInCBCoord[2]
            # cam_ground_proj_to_hp=math.sqrt(math.pow(CamCenterInCBCoord[0]-resulthp[0],2)+math.pow(CamCenterInCBCoord[1]-resulthp[1],2))
            # angle=math.atan2(cam_height, cam_ground_proj_to_hp)


            # height=ground_dist*math.tan(angle)
            
            # Recover the gps coordinate
            start = time.time()
            gps_coord = RecoverGPSCoord(resultg)
            #print("One object recover gps took {} seconds.".format(time.time() - start))
            #print("one bb localization took {} seconds.".format(time.time() - start))
            #print("gps_coord: ", gps_coord)   
            objecta['gps']=gps_coord
            object_array.append(objecta)
            #print("One object handling took {} seconds.".format(time.time() - starto))

    start = time.time()
    content={}
    timestamp = str(datetime.datetime.now())
    content['timestamp']=timestamp
    content['frame_size']={'width': vw, 'length': vh}
    content['average_fps']=20
    content['safety_region_coord']=regionpts.tolist()
    content['object_array']=object_array
    #content['frame_in_b64']='data:image/jpeg;base64,' + urllib.parse.quote(base64.b64encode(buffer))
    #print("size of frame: ", len(content['frame_in_b64']))
    #filew.write(json.dumps(content))
    #producer.send('mwc_camera_info', content)
    print("send out in kafka took {} seconds.".format(time.time() - start))

    #cv2.imshow('Stream', frame)
    #outvideo.write(frame)
    # ch = 0xFF & cv2.waitKey(1)
    # if ch == 27:
    #     break

    print("One loop took {} seconds.".format(time.time() - realstart))

#filew.close()
totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
#cv2.destroyAllWindows()
#outvideo.release()

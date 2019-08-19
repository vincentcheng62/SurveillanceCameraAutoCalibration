from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
#import tensorflow as tf

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
conf_thres_global=0.1 # higher means less object detect
conf_thres_zoom=0.05 # higher means less object detect
nms_thres=0.3 #higher means less bb are going to be rejected. original 0.4
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


def round_to_1(x, sig=2):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)

def detect_image(img, IsUsingZoomInDetection):
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

    if IsUsingZoomInDetection:
        conf_thres = conf_thres_zoom
    else:
        conf_thres = conf_thres_global

    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


fs_read = cv2.FileStorage("homography.yml", cv2.FILE_STORAGE_READ)
homography_matrix = fs_read.getNode("homography_matrix").mat()
print("homography_matrix: ", homography_matrix)
fs_read.release()  

fs_read = cv2.FileStorage("intrinsic_and_distortion_coeff.yml", cv2.FILE_STORAGE_READ)
camera_matrix_manual = fs_read.getNode("intrinsic").mat()
print("intrinsic: ", camera_matrix_manual)
dist_coefs_manual = fs_read.getNode("distortion_coeff").mat()
print("distortion_coeff: ", dist_coefs_manual)
fs_read.release()  

IsOutputVideo=True
IsSavingZoomAreaRawVideo=False
IsUsingZoomInDetection=False

#videopath = '../data/video/overpass.mp4'
#videopath = 'IMG_6308.MOV'
#videopath = 'traffic2_trim.mp4'
#videopath = 'test_far_ppl3.mp4'
#videopath = 'test_far_ppl_4k_3.mp4'
#videopath = 'test_far_ppl_8_mount.mp4'
videopath = 'test_far_ppl_8_mount-detcut-zoom-in.mp4'
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

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)

outvideo=None
outvideo2=None
save_video_fps=15.0


#ZID: Zoom in detection, so as to detect far object


# for test_far_ppl3.mp4
# ZID_x=675
# ZID_y=300
# ZID_w=425
# ZID_h=425

# for test_far_ppl_4k.mp4
# ZID_x=1530
# ZID_y=600
# ZID_w=608
# ZID_h=608

# for dji osmo pocket
# ZID_x=1630
# ZID_y=1000
# ZID_w=608
# ZID_h=608

# middle point is (1920, 1080)

# ZID_x=1530
# ZID_y=1000
# ZID_w=608
# ZID_h=608

ZID_x=1460
ZID_y=750
ZID_w=608
ZID_h=608

# ZID_x=1900
# ZID_y=1200
# ZID_w=304
# ZID_h=304

if IsOutputVideo:
    outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det-zzz.mp4"),fourcc,save_video_fps,(int(vw/2) ,int(vh/2)))
    if IsUsingZoomInDetection:
        print ("zoom in Video size", ZID_w,ZID_h)
        outvideo2 = cv2.VideoWriter(videopath.replace(".mp4", "-det-zoom-in.mp4"),fourcc,save_video_fps,(ZID_w,ZID_h))
    


frames = 0
counter=0
starttime = time.time()
regionpts = numpy.array([[85,109],[55,225],[582,236],[495,127]], numpy.int32)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

#while(counter<total_frame):
while(True):

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
    detections = detect_image(pilimg, False)

    print("global detection number: ", len(detections))

    zoom_frame=None
    if IsUsingZoomInDetection:
        zoom_frame = frame[ZID_y:ZID_y+ZID_h, ZID_x:ZID_x+ZID_w]
        zoom_pilimg = Image.fromarray(zoom_frame.copy())        
        zoom_detections = []
        zoom_detections = detect_image(zoom_pilimg, IsUsingZoomInDetection)

        if zoom_detections is not None:
            print("zoom in detection number: ", len(zoom_detections))

            for q in range(0, len(zoom_detections)):
                zoom_detections[q][0] = zoom_detections[q][0] + ZID_x
                zoom_detections[q][1] = zoom_detections[q][1] + ZID_y
                zoom_detections[q][2] = zoom_detections[q][2] + ZID_x
                zoom_detections[q][3] = zoom_detections[q][3] + ZID_y

            detections = torch.cat((detections, zoom_detections), 0)
            
        print("Total detection number: ", len(detections))

        zoom_frame = cv2.cvtColor(zoom_frame, cv2.COLOR_RGB2BGR)

    print("convert color and run detection took {} seconds.".format(time.time() - start))

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    #retval, buffer = cv2.imencode('.jpeg', frame, encode_param)

    object_array=[]

    # draw the zoom in detection area
    if IsUsingZoomInDetection:
        cv2.rectangle(frame, (ZID_x, ZID_y), (ZID_x+ZID_w, ZID_y+ZID_h), (255,0,0), 4)
        cv2.putText(frame, "zoom-in detection area" , (ZID_x+ZID_w-350, ZID_y+ZID_h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 8)

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
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]

            if IsOutputVideo:
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

                if IsUsingZoomInDetection:
                    if not IsSavingZoomAreaRawVideo:
                        cv2.rectangle(zoom_frame, (x1-ZID_x, y1-ZID_y), (x1+box_w-ZID_x, y1+box_h-ZID_y), color, 4)
                    # cv2.rectangle(zoom_frame, (x1-ZID_x, y1-35-ZID_y), (x1+len(cls)*19+80-ZID_x, y1-ZID_y), color, -1)
                    # cv2.putText(zoom_frame, cls + "-" + str(int(obj_id)), (x1-ZID_x, y1 - 10 -ZID_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

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
            bottom_pt = [(xA+xB)*0.5, max(yA, yB)] # (u2, v2)
            #print("bottom_pt", bottom_pt)

            bp = np.zeros((1, 2), np.float32)
            bp[0][0] = bottom_pt[0]
            bp[0][1] = bottom_pt[1]
            dst = cv.undistortPoints(bp.reshape(-1,1,2).astype(np.float64), camera_matrix_manual, dist_coefs_manual)
            #print("after undistort: ", dst[0][0])

            gps_coord = cv2.perspectiveTransform(dst, homography_matrix)
            #print("gps_coord: ", gps_coord[0][0])
            objecta['gps']=gps_coord[0][0]
            object_array.append(objecta)
            #print("One object handling took {} seconds.".format(time.time() - starto))

            if IsOutputVideo:
                cv2.putText(frame, "(" + str(gps_coord[0][0][0]) + ", " + str(gps_coord[0][0][1]) + ")", (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                # if IsUsingZoomInDetection:
                #     cv2.putText(zoom_frame, "(" + str(gps_coord[0][0][0]) + ", " + str(gps_coord[0][0][1]) + ")", (x1-ZID_x, y1+20-ZID_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)                    

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
    if IsOutputVideo:
        half_frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5) 
        outvideo.write(half_frame)
        if IsUsingZoomInDetection:
            outvideo2.write(zoom_frame)
    # ch = 0xFF & cv2.waitKey(1)
    # if ch == 27:
    #     break

    print("One loop took {} seconds.".format(time.time() - realstart))

#filew.close()
totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
#cv2.destroyAllWindows()

if IsOutputVideo:
    outvideo.release()
    if IsUsingZoomInDetection:
        outvideo2.release()

#!/usr/bin/python

import cv2
import numpy as np
import sys

# img1 = cv2.imread('maskout.bmp')
# img1 = img1.astype(np.float32)
# img2 = cv2.imread('maskout_-15.bmp')
# img2 = img2.astype(np.float32)

img1 = cv2.imread('IMG_7094_half.JPG')
img1 = img1.astype(np.float32)
img2 = cv2.imread('screen_1920x1080_4.png')
img2 = img2.astype(np.float32)

mapper = cv2.reg_MapperGradAffine()
mappPyr = cv2.reg_MapperPyramid(mapper)

resMap = mappPyr.calculate(img1, img2)
mapShift = cv2.reg.MapTypeCaster_toAffine(resMap)

print(mapShift.getLinTr())
print(mapShift.getShift())

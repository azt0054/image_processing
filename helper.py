import cv2
import os
import numpy as np
import datetime
import math

from config import *

def dStack(*images):
    stacked = []
    for im in images:
        if len(im.shape)>2:
            stacked.append(im)
        else:
            stacked.append(np.dstack([im]*3))
    return stacked

def hStack(*images):
    return np.hstack(dStack(*images))

def vStack(*images):
    return np.vstack(dStack(*images))

def get2DFrom3D(image):
    try:
        image_2d = cv2.cvtColor(bottom_mask,cv2.COLOR_BGR2GRAY)
    except:
        filename = "temp_cv_%s.jpg"%datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        cv2.imwrite(filename,image)
        image_2d = cv2.imread(filename,0)
        os.remove(filename)
    return image_2d

def dilate(image, iter_count=1, kernel=kernel_size):
    dilated = cv2.dilate(image, np.ones(kernel,np.uint8), iterations=iter_count)
    return dilated

def erode(image, iter_count=1, kernel=kernel_size):
    eroded = cv2.erode(image, np.ones(kernel,np.uint8), iterations=iter_count)
    return eroded

def morphOpen(image, iter_count=1, kernel=kernel_size):
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones(kernel,np.uint8), iterations=iter_count)
    return opened

def morphclose(image, iter_count=1, kernel=kernel_size):
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones(kernel,np.uint8), iterations=iter_count)
    return closed

def adaptiveThresh(image,threshold_block_size=threshold_block_size):
    adaptive_thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold_block_size, 1)
    return adaptive_thresholded

def thresh(image):
    ret,thresholded = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    return ret,thresholded

def addText(image, text):
    cv2.putText(image, "{}".format(text), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

def show(image,wait_time=0):
    cv2.imshow("CVFilter", image)
    k = cv2.waitKey(wait_time)
    if k == 27:
        cv2.destroyAllWindows()

def resize(image,background_color=background_color):
    height, width = output_image_size
    resized_image = np.zeros((height,width,3), np.uint8)
    resized_image[resized_image==0] = background_color

    current_height, current_width = image.shape[:2]

    diff_height = height - current_height
    diff_width = width - current_width

    if diff_height<diff_width:
        current_ratio = current_width/float(current_height)
        new_height = int(height-resize_margin*2)
        new_width = int(height*(current_ratio))
        y_offset = resize_margin
        x_offset = (width - new_width)/2
    else:
        current_ratio = current_height/float(current_width)
        new_width = int(width-resize_margin*2)
        new_height = int(width*(current_ratio))
        x_offset = resize_margin
        y_offset = (height - new_height)/2

    resized_with_aspect_ratio = cv2.resize(image, (new_width,new_height))
    resized_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_with_aspect_ratio
    return resized_image

def getHSV(image):
    height, width = image.shape[:2]
    min_hsv = [179,255,255]
    max_hsv = [0,0,0]

    boundry = [
        image[0:corner_size, 0:corner_size],
        image[height-corner_size:height, 0:corner_size],
        image[0:corner_size, width-corner_size:width],
        image[height-corner_size:height, width-corner_size:width],
    ]

    for roi in boundry:
        hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        roi_min = hsv.min(1).min(0)
        roi_max = hsv.max(1).max(0)

        for index in range(len(min_hsv)):
            if roi_min[index] < min_hsv[index]:
                min_hsv[index] = roi_min[index]
            if roi_max[index] > max_hsv[index]:
                max_hsv[index] = roi_max[index]

    return min_hsv, max_hsv

def checkDir(*directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def log(*log_string):
    log_string = " ".join(log_string)
    print log_string

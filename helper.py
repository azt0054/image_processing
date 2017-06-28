import cv2
import os
from functools import wraps
import errno
import signal
from skimage.measure import compare_ssim as ssim
import numpy as np
import datetime
import math
import random

from config import *


border_map = {0:'TL',1:'TR',2:'BR',3:'BL',4:'L',5:'T',6:'R',7:'B'}
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
        image_2d = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    except:
        rand_int = random.randrange(100000,999999)
        filename = "temp_cv_%s%s.jpg"%(datetime.datetime.now().strftime("%Y%m%d%H%M%S"),str(rand_int))
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

def thresh(image,min_value = 127):
    ret,thresholded = cv2.threshold(image,min_value,255,cv2.THRESH_BINARY_INV)
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

def getBorderInfo(image):
    height, width = image.shape[:2]

    color_info = {}
    corners = [
        image[0:corner_size, 0:corner_size],
        image[0:corner_size, width-corner_size:width],
        image[height-corner_size:height, width-corner_size:width],
        image[height-corner_size:height, 0:corner_size],
    ]

    edges=[]
    edge_hsv_info = []
    if edge_depth:
        row_mid = height/2
        col_mid = width/2
        edges = [
            image[row_mid:row_mid+corner_size, 0:edge_depth],
            image[0:edge_depth, col_mid:col_mid+corner_size],
            image[row_mid:row_mid+corner_size, width-edge_depth:width],
            image[height-edge_depth:height, col_mid:col_mid+corner_size],
        ]
        for edge in edges:
            edge_hsv = getHSVOnROI([edge])
            edge_hsv_info.append(edge_hsv)


    min_hsv, max_hsv = getHSVOnROI(corners)
    color_info = getColorInfo(corners+edges)

    return min_hsv, max_hsv, edge_hsv_info, color_info

def getHSVOnROI(boundry):
    min_hsv = [179,255,255]
    max_hsv = [0,0,0]

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

def getColorInfo(boundry):
    color_info = {}
    for i,roi in enumerate(boundry):
        roi_min = roi.min(1).min(0)
        roi_max = roi.max(1).max(0)
        color_info[border_map[i]] = {}
        color_info[border_map[i]]['color'] = getColor(roi_min)
        color_info[border_map[i]]['min'] = roi_min
        color_info[border_map[i]]['max'] = roi_max
    return color_info

def getColor(roi):
    color = ""
    major_color = 0
    color_dict = {0:'blue',1:'green',2:'red'}
    for i,c in enumerate(roi):
        if c>major_color:
            color = color_dict[i]
            major_color = c
        elif c==major_color:
            color += color_dict[i]
    color = (color=='bluegreenred' and 'white') or color
    return color

def isSimilar(image,target_image):
    if image.shape == target_image.shape:
        probablity = ssim(image,target_image)
        if 0:
            image_random = np.random.rand(100,100)
            target_random = np.random.rand(100,100)
            image_norm = image_random/np.sqrt(np.sum(image_random**2))
            target_image_norm = target_random/np.sqrt(np.sum(target_random**2))
            probablity = np.sum(image_norm*target_image_norm)
        return probablity>0.85
    else:
        return False

def isDefaultImage(image):
    target_image = cv2.imread(os.path.join(image_root_dir,"default.jpg"),0)
    return isSimilar(image,target_image)


def customFill(image,origin=(0,0)):
    floodfill_image = image.copy()
    # Mask used to flood filling.
    rect_height, rect_width = image.shape[:2]
    fmask = np.zeros((rect_height, rect_width), np.uint8)
    fmask = vStack(image[0][:],image,image[-1][:])
    fmask = hStack(fmask[:,[0]][:],fmask,fmask[:,[-1]][:])
    fmask = get2DFrom3D(fmask)

    origin = (0,0) if origin==(0,0) else (rect_width-1,0) if origin==(0,1) else (0,rect_height-1) if origin==(1,0) else (rect_width-1,rect_height-1)
    cv2.floodFill(floodfill_image, fmask, origin, (255,255,255))
    return floodfill_image


def checkDir(*directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def env(env_name):
    return os.environ.get(env_name)

def log(*log_string):
    log_string = " ".join(log_string)
    print log_string

def timeout(seconds=3, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise ValueError("timeout")

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

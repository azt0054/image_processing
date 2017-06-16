import cv2
import numpy as np
from helper import *
from config import *
from contour_tools import *

def getSmallestRect(image, margin=10):
    image_height, image_width = image.shape[:2]

    origin_x,origin_y,rect_width,rect_height = cv2.boundingRect(image)

    new_origin_x,new_origin_y = origin_x-margin if origin_x-margin>0 else 1,origin_y-margin if origin_y-margin>0 else 1
    adj_rect_width,adj_rect_height = rect_width + (origin_x - new_origin_x),rect_height + (origin_y - new_origin_y)

    if margin:
        if rect_width/float(rect_height) < 0.85:
            new_rect_width,new_rect_height = adj_rect_width + int(margin*1.5),adj_rect_height + int(margin*2.0)
        elif rect_height/float(rect_width) < 0.8:
            new_rect_width,new_rect_height = adj_rect_width + int(margin*1.5),adj_rect_height + int(margin*1)
        else:
            new_rect_width,new_rect_height = adj_rect_width + int(margin*1),adj_rect_height + int(margin*1.25)

        rect_width = new_rect_width if new_rect_width<(image_width-new_origin_x) else max(adj_rect_width+margin if adj_rect_width+margin<image_width else adj_rect_width, image_width-new_origin_x-int(margin*0.5))
        rect_height = new_rect_height if new_rect_height<(image_height-new_origin_y) else max(adj_rect_height+margin if adj_rect_height+margin<image_height else adj_rect_height, image_height-new_origin_y-int(margin*0.5))

    image_cp = image.copy()
    cv2.rectangle(image_cp,(new_origin_x,new_origin_y),(new_origin_x+rect_width,new_origin_y+rect_height),(255,255,255),3)
    return new_origin_x,new_origin_y,rect_width,rect_height,image_cp

def createRectCuttingMask(image,x,y,width,height):
    ht, wd = image.shape[:2]
    blank_image = np.zeros((ht,wd,3), np.uint8)
    blank_image[:] = (255,255,255)
    cutting_mask = cv2.rectangle(blank_image,(x,y),(x+width,y+height),(0,0,0),-1)
    return cutting_mask

def getRectMask(image):
    x,y,wd,ht,new_image = getSmallestRect(image,0)
    mask = createRectCuttingMask(new_image,x,y,wd,ht)
    return mask

def getSmallestConvexMask(image,contour_size=min_contour_size,add_margin=0,contours=None):
    if not contours:
        _,contours,ehiers = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    if contour_size:
        contours = simpleContourFilter(contours, contour_size)

    clubbed_contours = np.vstack(contours[i] for i in range(len(contours)))
    hull = cv2.convexHull(clubbed_contours)
    (hull_count,_,_) = hull.shape
    hull.ravel()
    hull.shape = (hull_count,2)
    polyline = np.zeros_like(image)
    cv2.polylines(polyline,np.int32([hull]),True,(255,255,255))
    cv2.fillConvexPoly(polyline,np.int32([hull]),(255,255,255))
    if add_margin:
        polyline = dilate(polyline,add_margin)
    ret_polyline, reverse_polyline = cv2.threshold(polyline,127,255,cv2.THRESH_BINARY_INV)
    return ret_polyline, reverse_polyline

def applyMaskRules(mask,mask_info,probables=False):
    if mask_info.get("sure_foreground",None) is not None:
        sure_foreground = mask_info["sure_foreground"]
        mask = np.where((sure_foreground==255),1,mask).astype('uint8')
    if probables and mask_info.get("probable_foreground",None) is not None:
        probable_foreground = mask_info["probable_foreground"]
        mask = np.where((probable_foreground==255),3,mask).astype('uint8')
    if mask_info.get("sure_background",None) is not None:
        sure_background = erode(mask_info["probable_foreground"],4)
        mask = np.where((sure_background==255),0,mask).astype('uint8')
    return mask

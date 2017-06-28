import os
import sys
import cv2
import numpy as np

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from config import *
from helper import *
from mask_tools import *
from contour_tools import *


def checkDefects(image,contour,defects):
    image_cp = image.copy()
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        cv2.line(image_cp,start,end,[0,255,0],2)
        cv2.circle(image_cp,far,5,[0,0,255],-1)
    show(image_cp,5000)

def checkValidMask(mask_image):
    _,contours,ehiers = cv2.findContours(mask_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    contours = getSortedContourInfo(contours)[:1]
    clubbed_contours = np.vstack(contours[i] for i in range(len(contours)))

    hull_points = cv2.convexHull(contours[0], returnPoints = False)
    defects = cv2.convexityDefects(contours[0],hull_points)
    checkDefects(mask_image,contours[0],defects)

    hull = cv2.convexHull(clubbed_contours)
    (hull_count,_,_) = hull.shape
    hull.ravel()
    hull.shape = (hull_count,2)

    polyline = np.zeros_like(mask_image)
    cv2.polylines(polyline,np.int32([hull]),True,(255,255,255))
    cv2.fillConvexPoly(polyline,np.int32([hull]),(255,255,255))
    show(hStack(polyline,mask_image),5000)

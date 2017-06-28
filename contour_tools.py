import cv2
import numpy as np
from helper import *
from config import *

def getContours(image,contour_type=1):
    _,contours,ehiers = cv2.findContours(image,contour_type,2)
    return contours

def simpleContourFilter(contours, min_area):
    filtered_contours = filter(lambda cnt: cv2.contourArea(cnt)>min_area, contours)
    return filtered_contours

def getSortedContourInfo(contours):
    contour_info = []
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))

    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    return zip(*contour_info)[0]

def getCenterOfContour(contour):
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00']) if M['m00'] else int(M['m10']/0.1)
    cy = int(M['m01']/M['m00']) if M['m00'] else int(M['m01']/0.1)
    return np.array((cx,cy))

def getCenterContour(image,contours):
    height, width = image.shape[:2]
    center = np.array((width/2,height/2))
    contour_info = []
    for c in contours:
        dist = np.linalg.norm(center-getCenterOfContour(c))
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
            dist
        ))
    contour_info = sorted(contour_info, key=lambda c: (c[2], -c[3]), reverse=True)
    return zip(*contour_info)[0]


def getContourFromImage(image):
    contours = getContours(image)
    sorted_contour = getSortedContourInfo(contours)
    return sorted_contour

def getImageFromContours(image,contours):
    height, width = image.shape[:2]
    contour_image = np.zeros((height,width), np.uint8)
    cv2.drawContours(contour_image,contours,-1,(255,255,255),-1)
    return contour_image

def getContourApprox(contour,arc_length=0.1):
    epsilon = arc_length*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    return approx

def getExtremePoints(contour):
    leftmost = tuple(contour[contour[:,:,0].argmin()][0])
    rightmost = tuple(contour[contour[:,:,0].argmax()][0])
    topmost = tuple(contour[contour[:,:,1].argmin()][0])
    bottommost = tuple(contour[contour[:,:,1].argmax()][0])
    return leftmost, topmost, rightmost, bottommost

def distanceFilter(contours, max_contour_distance):
    if len(contours)>1:
        closed_contour = [contours[0],]
        open_contours = contours[1:]
        for opc in open_contours:
            for clc in closed_contour:
                status,distance_info = checkIfInside(clc,opc,min_inside=0,distance=True)
                if min(distance_info)>0:
                    closed_contour.append(opc)
                    break
                elif min(map(abs,distance_info))<max_contour_distance:
                    closed_contour.append(opc)
                    break
        return closed_contour
    else:
        return contours

def checkIfInside(polygon,contour,min_inside=3,distance=False):
    extreme_points = getExtremePoints(contour)
    is_inside = 0
    distance_info = []
    outside_distance = []
    inside_distance = []

    for pt in  extreme_points:
        distance = cv2.pointPolygonTest(polygon,pt,True)
        distance_info.append(distance)
        if distance>0:
            inside_distance.append(distance)
            is_inside += 1
        else:
            outside_distance.append(distance)

    if is_inside==4 or (min_inside and min_inside==is_inside==3 and outside_distance[0] > -30):
        return True if not distance else True,distance_info
    else:
        return False if not distance else False,distance_info

def findIfClose(cnt1,cnt2, distance=min_contour_distance):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < distance:
                return True
            elif i==row1-1 and j==row2-1:
                return False

def findCloserContour(grouped_contours,open_contours):
    open_contours_local = []
    for open_cnt in open_contours:
        is_closer = 0
        for grouped_cnt in grouped_contours:
            if findIfClose(grouped_cnt,open_cnt):
                is_closer =1
                break
        if is_closer:
            grouped_contours.append(open_cnt)
        else:
            open_contours_local.append(open_cnt)
    return grouped_contours, open_contours_local


def groupContours(contours,contour_limit=0,distance=2,size_flag=0):
    contour_info = getSortedContourInfo(contours)

    grouped_contours = [contour_info[0]]
    open_contours = contour_info[1:contour_limit]
    current_group_size = len(grouped_contours)

    while 1 and distance:
        grouped_contours,open_contours = findIfCloser(grouped_contours,open_contours,distance,size_flag)
        if len(grouped_contours)<=current_group_size:
            break
        else:
            current_group_size = len(grouped_contours)

    return grouped_contours if distance else contour_info

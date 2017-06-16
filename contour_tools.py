import cv2
import numpy as np

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

def getContourFromImage(image):
    contours = getContours(image)
    sorted_contour = getSortedContourInfo(contours)
    return sorted_contour

def getImageFromContours(image,contours):
    height, width = image.shape[:2]
    contour_image = np.zeros((height,width), np.uint8)
    cv2.drawContours(contour_image,contours,-1,(255,255,255),-1)
    return contour_image

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

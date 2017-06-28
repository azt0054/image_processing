import os
import sys
import cv2
import numpy as np

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from config import *
from helper import *

def autoCanny(image, sigma=0.33):
	v = np.median(image)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged

def getCannyEdge(image, threshold_limit=max_edge_threshold_limit):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edge = cv2.Canny(blurred, 0, threshold_limit)
    auto = autoCanny(image)
    return edge,auto

def cannyEdge(filename,threshold_limit):
	image = cv2.imread(filename)
	edge,auto = getCannyEdge(image,threshold_limit)
	edge2,auto = getCannyEdge(image,20)
	edge3,auto = getCannyEdge(image,10)
	stacked_image = hStack(edge,edge2,edge3)
	show(stacked_image)

if __name__ == '__main__':
	filename = '../images/original/12545.jpg'
	cannyEdge(filename,90)

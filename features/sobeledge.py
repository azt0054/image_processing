import cv2
import numpy as np
import sys
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from helper import *
from config import *
from contour_tools import *
from mask_tools import *
import cannyedge as ce

def sobelEdge(filename):
    image = cv2.imread(filename,0)

    # Convert to HSV for simpler calculations
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calcution of Sobelx
    sobelx = cv2.Sobel(image,cv2.CV_8U,1,0,ksize=3)
    abs_sobelx64f = np.absolute(sobelx)
    sobelx_8u = np.uint8(abs_sobelx64f)

    # Calculation of Sobely
    sobely = cv2.Sobel(image,cv2.CV_8U,0,1,ksize=3)
    abs_sobely64f = np.absolute(sobely)
    sobely_8u = np.uint8(abs_sobely64f)

    # Calculation of Laplacian
    laplacian = cv2.Laplacian(image,cv2.CV_8U,ksize=3,scale=5,delta=1)

    sobel = morphOpen(sobelx,1,kernel=(1,1))+morphOpen(sobely,1,kernel=(1,1))

    basic_edge,auto = ce.getCannyEdge(sobel,max_edge_threshold_limit)
    x,y,wd,ht,new_image = getSmallestRect(basic_edge,0)
    full_mask = createRectCuttingMask(new_image,x,y,wd,ht)

    laplacian_masked = cv2.subtract(dStack(laplacian)[0],full_mask)

    return sobel,laplacian_masked
    res = hStack(sobel,laplacian,full_mask,laplacian_masked)
    #test_file = os.path.join(test_image_dir,filename.split('/')[-1])
    #cv2.imwrite(test_file,res)

    """
    cv2.imshow('edges',res)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    """


if __name__ == '__main__':
	filename = '../images/processed/background_images/17677.jpg'
	sobelEdge(filename)

import numpy as np
import cv2
import sys
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from config import *
from helper import *

def autoAdjustGamma(image, gamma_limit=gamma_limit):
	inv_gamma = 1.0 / gamma_limit
	table = np.array([((i / 255.0) ** inv_gamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	return cv2.LUT(image, table)

def getGamma(image,gamma_limit=gamma_limit):
	adjusted_gamma = autoAdjustGamma(image, gamma_limit)

	image_2d = get2DFrom3D(adjusted_gamma)
	adaptive_thresholded = adaptiveThresh(image_2d,255)
	ret, reverse_thresholded = thresh(adaptive_thresholded)
	return adjusted_gamma, reverse_thresholded

def gammaCorrection(filename,gamma_limit):
	image = cv2.imread(filename)
	adjusted_gamma, reverse_thresholded = getGamma(image,gamma_limit)
	stacked_image = hStack(adjusted_gamma, reverse_thresholded)
	show(stacked_image)

if __name__ == '__main__':
	filename = '../images/original/12545.jpg'
	gammaCorrection(filename,0.2)

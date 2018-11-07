import cv2
import numpy
import math

def compactness(contour):
	area = int(cv2.contourArea(contour))
	perimeter = cv2.arcLength(contour, True)

	comp = (perimeter**2)/area
	compNorm = 1 - (4*math.pi*area)/(perimeter**2)
	return compNorm
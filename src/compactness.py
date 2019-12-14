import cv2
import math


def compactness(contour):
	area = int(cv2.contourArea(contour))
	perimeter = cv2.arcLength(contour, True)

	comp_norm = 1 - (4*math.pi*area)/(perimeter**2)
	return comp_norm

import cv2
import numpy as np
import math

def findCircle(contour):
	(x,y), radius = cv2.minEnclosingCircle(contour)
	center = (int(x),int(y))
	radius = int(radius)
	return center
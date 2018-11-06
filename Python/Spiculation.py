import cv2
import numpy as np
import math
import importlib

imgLoader = importlib.import_module('ImgLoader')
name = 'SPIC_DB_207.jpg'

imgColor, imgGray = imgLoader.loadImg(name)
canvas = imgLoader.createClearCanvas(imgColor)
contours = imgLoader.preProcess(imgGray)
cnt = maxAreaContour(contours)

def maxAreaContour(contours):
	cnt = contours[0]
	max_area = cv2.contourArea(cnt)

	for cont in contours:
		if cv2.contourArea(cont) > max_area:
			cnt = cont
			max_area = cv2.contourArea(cont)
	return cnt

def slope(pt0, pt):
	x1 = pt0[0]
	y1 = pt0[1]
	print(x1, y1)

	x2 = pt[0]
	y2 = pt[1]
	print(x2, y2)

	slope = (y2 - y1) / (x2 - x1)
	return slope

def angle(line1, line2):
	pt0Ln1 = np.array([line1[0][0], line1[0][1]])
	ptLn1 = np.array([line1[1][0], line1[1][1]])

	pt0Ln2 = np.array([line2[0][0], line2[0][1]])
	ptLn2 = np.array([line2[1][0], line2[1][1]])

	slopeLn1 = slope(pt0Ln1, ptLn1)
	slopeLn2 = slope(pt0Ln2, ptLn2)

	tanTheta = math.fabs((slopeLn2 - slopeLn1)/(1 +(slopeLn2*slopeLn1)))
	thetaRandians = math.atan(tanTheta)
	thetaDegrees = math.degrees(thetaRandians)
	return thetaDegrees

perimeter = int(cv2.arcLength(cnt, False))
epsilon = 0.01*perimeter
approx = cv2.approxPolyDP(cnt, epsilon, False) #parâmetros para testar: epsilon(dita o quao simplificada fica a figura)
import cv2
import numpy as np
import math

def loadImg(name):
	img = cv2.imread(name)
	img = cv2.resize(img, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
	imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cv2.imshow("imagem bruta", img)
	cv2.waitKey(0)
	return img, imgGray

def createClearCanvas(img):
	canvas = np.zeros(img.shape, np.uint8)
	return canvas

def preProcess(imgGray):
	ret,thresh = cv2.threshold(imgGray,20,255,cv2.THRESH_BINARY) 
	cv2.imshow("binarizacao da imagem", thresh)
	cv2.waitKey(0)
	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	print(len(contours))
	print(contours)
	print(hierarchy)
	return contours

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

#def spiculationIndex(polygonalModel):

imgColor, imgGray = loadImg("SPIC_DB_207.jpg")
cv2.imshow("imagem tons de cinza", imgGray)
cv2.waitKey(0)
canvas = createClearCanvas(imgColor)
contours = preProcess(imgGray)
cnt = maxAreaContour(contours)

perimeter = int(cv2.arcLength(cnt, False))
epsilon = 0.01*perimeter
approx = cv2.approxPolyDP(cnt, epsilon, False) #parâmetros para testar: epsilon(dita o quao simplificada fica a figura)

height, width, depth = approx.shape
approx = approx.reshape(height,depth)
print(approx)

ln1 = np.array([approx[3], approx[4]])
ln2 = np.array([approx[2], approx[3]])
print(ln1)
print(ln2)
print(angle(ln1, ln2))

cv2.drawContours(canvas, [approx], 0, (0, 0, 255), 1)

cv2.imshow("contorno", canvas)
cv2.imwrite("teste_polygon.jpg",canvas)
cv2.waitKey(0)
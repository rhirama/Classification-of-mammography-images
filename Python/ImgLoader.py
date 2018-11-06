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
	ret,thresh = cv2.threshold(imgGray,20,255,cv2.THRESH_BINARY) #dependendo da imagem modificar os valores de thresh e maxval
	cv2.imshow("binarizacao da imagem", thresh)
	cv2.waitKey(0)
	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours


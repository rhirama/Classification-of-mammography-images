import cv2
import numpy as np
import math
import importlib

imgLoaderMod = importlib.import_module('ImgLoader')
spIndexMod = importlib.import_module('Spiculation')
compMod = importlib.import_module('Compactness')
findCircMod = importlib.import_module('Circle')

def maxAreaContour(contours):
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    return cnt

name = 'SPIC_DB_207.jpg'

imgColor, imgGray = imgLoaderMod.loadImg(name)
canvas = imgLoaderMod.createClearCanvas(imgColor)
contours = imgLoaderMod.preProcess(imgGray)
cnt = maxAreaContour(contours)

(x,y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(imgColor, center, radius, (0,255,0), 1)
cv2.imshow('circulo', imgColor)
cv2.waitKey(0)

perimeter = 2 * math.pi * radius #perimeter = int(cv2.arcLength(cnt, True))
epsilon = 0.01 * perimeter
approx = cv2.approxPolyDP(cnt, epsilon, True)# parâmetros para testar: epsilon(dita o quao simplificada fica a figura)
cv2.drawContours(canvas, [approx],  0, (0, 0, 255), 1)
cv2.imshow('modelo poligonal', canvas)
cv2.waitKey(0)

compNorm = compMod.compactness(approx)
si = spIndexMod.calcultateSpiculationIndex(approx)
print(compNorm)
print(si)
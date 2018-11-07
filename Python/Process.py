import cv2
import numpy as np
import math
import importlib

imgLoader = importlib.import_module('ImgLoader')
spIndex = importlib.import_module('Spiculation')

def maxAreaContour(contours):
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    return cnt

name = 'zikka.jpg'

imgColor, imgGray = imgLoader.loadImg(name)
canvas = imgLoader.createClearCanvas(imgColor)
contours = imgLoader.preProcess(imgGray)
cnt = maxAreaContour(contours)

perimeter = int(cv2.arcLength(cnt, False))
epsilon = 0.01*perimeter
approx = cv2.approxPolyDP(cnt, epsilon, False)# parâmetros para testar: epsilon(dita o quao simplificada fica a figura)
si = spIndex.calcultateSpiculationIndex(approx)
print(si)
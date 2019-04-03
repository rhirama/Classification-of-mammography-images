import cv2
import numpy as np
import math

im = cv2.imread("teste5.jpg")
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow("teste", imgray)
cv2.waitKey(0)
canvas = np.zeros(im.shape, np.uint8)
canvas2 = np.zeros(im.shape, np.uint8)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[1]

perimeter = int(cv2.arcLength(cnt, True))
epsilon = 0.01*perimeter
approx = cv2.approxPolyDP(cnt, epsilon, True)  # parâmetros para testar: epsilon(dita o quao simplificada fica a figura)

cv2.drawContours(canvas, [approx], 0, (0, 0, 255), 1)
cv2.drawContours(canvas2, [cnt], 0, (0, 255, 0), 1) 

cv2.imshow("teste", canvas)
cv2.imwrite("teste_polygon.jpg",canvas)
cv2.imwrite("teste_contorno.jpg",canvas2)
cv2.imshow("teste2", canvas2)
cv2.waitKey(0)


import numpy as np
import cv2

nome1 = 'C55_97LOROI.jpg'
nome2 = 'C62_97LCROI.jpg'
nome3 = 'CIRC_FB_069.jpg'
nome4 = 'SPIC_DB_193.jpg'

im = cv2.imread(nome4)
im = cv2.resize(im, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
#edges = cv2.Canny(im,75,100)
cv2.imshow('salve', im)
cv2.waitKey(0)

#imBen1 = cv2.imread('SPIC_DB_193.jpg')
canvas1 = np.zeros(im.shape, np.uint8)
canvas2 = np.zeros(im.shape, np.uint8)
img2gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

#kernel = np.ones((5,5),np.float32)/25
#img2gray = cv2.filter2D(img2gray,-1,kernel)

ret,thresh = cv2.threshold(img2gray,20,255,cv2.THRESH_BINARY)
cv2.imshow('andreblinha', thresh)
cv2.waitKey(0)
im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cnt = contours[0]

perimeter = cv2.arcLength(cnt,True)
epsilon = 0.001*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

cv2.drawContours(canvas2, [approx], 0, (0, 255, 0), 1)

cv2.imwrite('SPIC_DB_193_001perimetro.png', canvas2)


canvas2 = np.zeros(im.shape, np.uint8)

epsilon = 0.005*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

cv2.drawContours(canvas2, [approx], 0, (0, 255, 0), 1)

cv2.imwrite('SPIC_DB_193_005perimetro.png', canvas2)


canvas2 = np.zeros(im.shape, np.uint8)

epsilon = 0.01*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

cv2.drawContours(canvas2, [approx], 0, (0, 255, 0), 1)

cv2.imwrite('SPIC_DB_193_01perimetro.png', canvas2)

cv2.imshow('teste', canvas2)


cv2.waitKey(0)
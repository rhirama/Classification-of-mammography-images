from PIL import Image
import cv2
import numpy
import math

im = cv2.imread("teste3.jpg")
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[1]

height, width, channels = im.shape
perimeterM = 0

for i in range(height):
	for j in range(width):
		if numpy.all(im[i,j] <= (8,8,8)):
			if numpy.all(im[i+1,j] == (255,255,255)):
				perimeterM += 1 
			elif numpy.all(im[i-1,j] == (255,255,255)):
				perimeterM += 1 
			elif numpy.all(im[i,j+1] == (255,255,255)):
				perimeterM += 1 
			elif numpy.all(im[i,j-1] == (255,255,255)):
				perimeterM += 1 
			elif numpy.all(im[i+1,j+1] == (255,255,255)):
				perimeterM += 1 
			elif numpy.all(im[i-1,j+1] == (255,255,255)):
				perimeterM += 1 
			elif numpy.all(im[i+1,j-1] == (255,255,255)):
				perimeterM += 1 
			elif numpy.all(im[i-1,j-1] == (255,255,255)):
				perimeterM += 1 


cv2.drawContours(im2, [cnt], 0, (0, 255, 0), 3) 
cv2.imshow("teste", im2)
cv2.waitKey(0)
		
area = int(cv2.contourArea(cnt))
perimeter = cv2.arcLength(cnt, True)

def compactness(area, perimeter):
	comp = (perimeter**2)/area
	compNorm = 1 - (4*math.pi*area)/(perimeter**2)

	print("the compactness feature is " + str(comp))
	print("the normalized compactness feature is " + str(compNorm))
	print("")

print ("the area is " + str(area))
print ("the perimeter is " + str(perimeter))
print ("the perimeterM is " + str(perimeterM))
compactness(area, perimeterM)
compactness(area, perimeter)
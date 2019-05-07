import cv2
import numpy as np
from PIL import Image


def max_area_contour(contours):
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    return cnt


multiplier = 0.01
img = cv2.imread('C62_97LAXMROI.jpg')
canvas = np.zeros(img.shape, np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 20, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = max_area_contour(contours)
perimeter = int(cv2.arcLength(cnt, True))
epsilon = multiplier * perimeter
approx = cv2.approxPolyDP(cnt, epsilon, True)
cv2.drawContours(canvas, [approx], 0, (255, 255, 255), 1)
pinto = cv2.convexHull(approx)
cv2.drawContours(canvas, [pinto], 0, (255, 0, 255), 1)
cv2.imshow('img', canvas)
cv2.waitKey(0)

hull = cv2.convexHull(approx, returnPoints=False)
defects = cv2.convexityDefects(approx, hull)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(canvas, start, end, (0, 255, 0), 2)
    cv2.circle(canvas, far, 5, (0, 0, 255), -1)

cv2.imshow('img', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

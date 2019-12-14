import cv2
import importlib
import numpy as np
import glob
import os
import xlsxwriter

img_loader_mod = importlib.import_module('img_loader')
si_mod = importlib.import_module('spiculation')


def max_area_contour(contours):
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    return cnt


name = 'teste3.jpg'
multiplier = 0.01

img_color, img_gray = img_loader_mod.load_img(name)
canvas = img_loader_mod.create_clear_canvas(img_gray)
contours = img_loader_mod.pre_process(img_gray)
cnt = max_area_contour(contours)

perimeter = int(cv2.arcLength(cnt, True))
epsilon = multiplier * perimeter

approx = cv2.approxPolyDP(cnt, epsilon, True)  # parâmetros para testar: epsilon(dita o quao simplificada fica a figura)
cv2.drawContours(canvas, [approx],  0, (128, 128, 128), 1)
print(approx)
cv2.imshow('poly', canvas)
cv2.waitKey(0)
si, lista = si_mod.calculate_si(approx)

cv2.imshow('img', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


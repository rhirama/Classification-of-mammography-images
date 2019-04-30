import cv2
import numpy as np
import importlib


def max_area_contour(contours):
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    return cnt


def frac_concavity(approx):
    hull = cv2.convexHull(approx, clockwise=False)
    aux = 0

    for i in range(approx.shape[0]):
        if approx[i] == hull[aux]:
            aux += 1
        else:
            continue
    return hull


img_loader_mod = importlib.import_module('img_loader')

name = 'CIRC_FM_141.jpg'
multiplier = 0.01

img_color, img_gray = img_loader_mod.load_img(name)
canvas = img_loader_mod.create_clear_canvas(img_color)
contours = img_loader_mod.pre_process(img_gray)
cnt = max_area_contour(contours)

perimeter = int(cv2.arcLength(cnt, True))
epsilon = multiplier * perimeter
approx = cv2.approxPolyDP(cnt, epsilon, True)  # parâmetros para testar: epsilon(dita o quao simplificada fica a figura)
cv2.drawContours(canvas, [approx],  0, (255, 255, 255), 1)
hull = frac_concavity(approx)
cv2.drawContours(canvas, [hull],  0, (255, 0, 255), 1)

cv2.imshow('modelo poligonal', canvas)
cv2.waitKey(0)
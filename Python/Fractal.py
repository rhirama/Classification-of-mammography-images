import cv2
import importlib
import glob
import os
import xlsxwriter


img_loader_mod = importlib.import_module('ImgLoader')


def max_area_contour(contours):
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    return cnt


name = 'Sierpinski.jpg'
img = cv2.imread(name)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canvas = img_loader_mod.create_clear_canvas(img_gray)
cv2.imshow('imagem cinza', img_gray)



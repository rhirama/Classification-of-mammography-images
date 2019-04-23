import cv2
import importlib
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


name = 'SPIC_DB_207.jpg'
multiplier = 0.001

img_color, img_gray = img_loader_mod.load_img(name)
canvas = img_loader_mod.create_clear_canvas(img_color)
contours = img_loader_mod.pre_process(img_gray)
cnt = max_area_contour(contours)

perimeter = int(cv2.arcLength(cnt, True))
epsilon = multiplier * perimeter
approx = cv2.approxPolyDP(cnt, epsilon, True)  # parâmetros para testar: epsilon(dita o quao simplificada fica a figura)
cv2.drawContours(canvas, [approx],  0, (255, 255, 255), 1)

cv2.imshow('modelo poligonal', canvas)
cv2.waitKey(0)

si = si_mod.calculate_si(approx, perimeter)
print(si)


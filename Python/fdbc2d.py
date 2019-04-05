import cv2
import importlib
import glob
import os
import xlsxwriter
import pandas

img_loader_mod = importlib.import_module('img_loader')
fd_mod = importlib.import_module('fractal')


def max_area_contour(contours):
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    return cnt


name = 'SPIC_DB_207.jpg'
img_color, img_gray = img_loader_mod.load_img(name)
canvas = img_loader_mod.create_clear_canvas(img_gray)
contours = img_loader_mod.pre_process(img_gray)
cnt = max_area_contour(contours)

perimeter = int(cv2.arcLength(cnt, True))
epsilon = 0.001 * perimeter
approx = cv2.approxPolyDP(cnt, epsilon,
                                      True)  # parâmetros para testar: epsilon(dita o quao simplificada fica a figura)
cv2.drawContours(canvas, [approx], 0, (255, 255, 255), 1)

# ret, poly_bin = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
print(canvas.shape)
cv2.imshow('s', canvas)
cv2.waitKey(0)

print(fd_mod.fractal_dimension(canvas))

import cv2
import importlib
import glob
import os
import xlsxwriter
import pandas
from matplotlib import pyplot as plt



def max_area_contour(contours):
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    return cnt


img_loader_mod = importlib.import_module('img_loader')
fd_mod = importlib.import_module('Fractal')

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
im2, contours, hierarchy = cv2.findContours(canvas, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # find contours of canvas
cnt = max_area_contour(contours)
# ret, poly_bin = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
print(canvas.shape)
plt.imshow(canvas, cmap = 'binary', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

print(fd_mod.ruler_fractal_dimension(cnt))

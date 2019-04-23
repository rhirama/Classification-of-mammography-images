import cv2
import importlib
import glob
import os
import openpyxl
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


xlsx_54BND = 'D:\\Users\\Rodrigo S. Hirama\\Documentos\\EACH\\IC\\Classification-of-mammography-images\\Comparacao_contours54BND.xlsx'
xlsx_57EDG = 'D:\\Users\\Rodrigo S. Hirama\\Documentos\\EACH\\IC\\Classification-of-mammography-images\\Comparacao_contours57EDG.xlsx'

imgs_54BND = 'D:/Users/Rodrigo S. Hirama/Imagens/Contours54BND/*.jpg'
imgs_57EDG = 'D:/Users/Rodrigo S. Hirama/Imagens/Contours57EDG/*.jpg'

files_paths = [(xlsx_54BND, imgs_54BND), (xlsx_57EDG, imgs_57EDG)]
sheet_names = [0.05, 0.01, 0.001]
writer = None
features = None
fd = []


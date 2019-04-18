import cv2
import importlib
import glob
import os
import openpyxl
import xlsxwriter
import pandas
import numpy as np
from matplotlib import pyplot as plt


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

for file, img_path in files_paths:
        for multiplier in sheet_names:
            fd = []
            features = pandas.read_excel(file, sheet_name=str(multiplier), header=0, skipfooter=0)
            for name in glob.glob(img_path):
                img_color, img_gray = img_loader_mod.load_img(name)
                ret, thresh = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
                # RETR_EXTERNAL for getting only the outer contour and CHAIN_APPROX_NONE to return a list of contour points
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnt = max_area_contour(contours)

                cnt = np.squeeze(cnt)
                one_d = img_loader_mod.make_1d_contour(cnt)

                # make 1d image and save it on a temporary png
                fig = plt.figure(frameon=False)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.plot(one_d)
                fig.savefig('temp')

                # load the temp png and execute the ruler method
                img_color, img_gray = img_loader_mod.load_img("temp.png")
                img_gray = cv2.bitwise_not(img_gray)  # invert colors on the temp png
                ret, thresh = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
                # RETR_EXTERNAL for getting only the outer contour and CHAIN_APPROX_NONE to return a list of contour points
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnt = max_area_contour(contours)

                fd.append(fd_mod.ruler_fractal_dimension(cnt))
                name = os.path.basename(name)

            # fd_df = pandas.DataFrame(fd)
            features.insert(4, 'fd_1Druler', fd, True)
            # file = os.path.basename(file)
            excel = openpyxl.load_workbook(file, read_only=False)
            writer = pandas.ExcelWriter(file, engine='openpyxl')
            writer.book = excel
            writer.sheets = dict((ws.title, ws) for ws in excel.worksheets)
            features.to_excel(writer, sheet_name=str(multiplier), index=False)
            writer.save()
writer.close()

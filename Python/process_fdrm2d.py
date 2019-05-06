import cv2
import importlib
import glob
import os
import openpyxl
import xlsxwriter
import pandas

img_loader_mod = importlib.import_module('img_loader')
fd_mod = importlib.import_module('Fractal')


def max_area_contour(contours):
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    return cnt


xlsx_54BND = '/home/who/Documents/cg/Classification-of-mammography-images/Python/Comparacao_contours54BND.xlsx'
xlsx_57EDG = '/home/who/Documents/cg/Classification-of-mammography-images/Python/Comparacao_Contours57EDG.xlsx'

imgs_54BND = '/home/who/Documents/cg/Classification-of-mammography-images/Imagens/Contours54BND/*.jpg'
imgs_57EDG = '/home/who/Documents/cg/Classification-of-mammography-images/Imagens/Contours57EDG/*.jpg'

files_paths = [(xlsx_54BND, imgs_54BND), (xlsx_57EDG, imgs_57EDG)]
sheet_names = [0.01, 0.001]
writer = None
features = None
fd = []

for file, img_path in files_paths:
    for multiplier in sheet_names:
        fd = []
        features = pandas.read_excel(file, sheet_name=str(multiplier), header=0, skipfooter=0)
        for name in glob.glob(img_path):
            print(name)
            img_color, img_gray = img_loader_mod.load_img(name)
            canvas = img_loader_mod.create_clear_canvas(img_gray)
            ret, thresh = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
            # RETR_EXTERNAL for getting only the outer contour and CHAIN_APPROX_NONE to return a list of contour points
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnt = max_area_contour(contours)

            # make polygon approx.
            perimeter = int(cv2.arcLength(cnt, True))
            epsilon = multiplier * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon,
                                      True)  # parâmetros para testar: epsilon(dita o quao simplificada fica a figura)
            cv2.drawContours(canvas, [approx], 0, (255, 255, 255), 1)  # write polygon approx on canvas
            im2, contours, hierarchy = cv2.findContours(canvas, cv2.RETR_LIST,
                                                   cv2.CHAIN_APPROX_NONE)  # find contours of canvas
            cnt = max_area_contour(contours)

            # # show contour for debugging
            # cv2.drawContours(img_color, cnt, -1, (255, 0, 255))
            # cv2.namedWindow('help', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('help', 1200, 1200)
            # cv2.imshow('help', img_color)
            # cv2.waitKey(0)
            # exit(0)

            # run the ruler method
            fd.append(fd_mod.ruler_fractal_dimension(cnt))
            name = os.path.basename(name)

        # fd_df = pandas.DataFrame(fd)
        features.insert(4, 'fd_2Druler', fd, True)
        # file = os.path.basename(file)
        excel = openpyxl.load_workbook(file, read_only=False)
        writer = pandas.ExcelWriter(file, engine='openpyxl')
        writer.book = excel
        writer.sheets = dict((ws.title, ws) for ws in excel.worksheets)
        features.to_excel(writer, sheet_name=str(multiplier), index=False)
        writer.save()
writer.close()

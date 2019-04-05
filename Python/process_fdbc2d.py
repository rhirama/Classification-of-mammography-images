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


files_paths = ['D:\\Users\\Rodrigo S. Hirama\\Documentos\\EACH\\IC\\Comparacao_contours54BND.xlsx', 'D:\\Users'
                                                                                                    '\\Rodrigo S. '
                                                                                                    'Hirama\\Documentos\\EACH\\IC\\Comparacao_contours54BND.xlsx']
sheet_names = [0.05, 0.01, 0.001]
writer = None
features = None
fd = []

for file in files_paths:
    for multiplier in sheet_names:
        features = pandas.read_excel(file, sheet_name=str(multiplier), header=0, skipfooter=1)

        for name in glob.glob("D:/Users/Rodrigo S. Hirama/Imagens/Contours54BND/*.jpg"):
            img_color, img_gray = img_loader_mod.load_img(name)
            canvas = img_loader_mod.create_clear_canvas(img_gray)
            contours = img_loader_mod.pre_process(img_gray)
            cnt = max_area_contour(contours)

            # (x, y), radius = cv2.minEnclosingCircle(cnt)
            # center = (int(x), int(y))
            # radius = int(radius)
            # cv2.circle(img_color, center, radius, (0, 255, 0), 1)
            # cv2.imshow('circulo', img_color)
            # cv2.waitKey(0)

            # perimeter = 2 * math.pi * radius
            perimeter = int(cv2.arcLength(cnt, True))
            epsilon = multiplier * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon,
                                      True)  # parâmetros para testar: epsilon(dita o quao simplificada fica a figura)
            cv2.drawContours(canvas, [approx], 0, (255, 255, 255), 1)
            # cv2.imshow('modelo poligonal', canvas)
            # cv2.waitKey(0)

            fd.append(fd_mod.fractal_dimension(canvas))
            name = os.path.basename(name)

        features[4] = fd
        file_name = os.path.basename(file)
        excel = openpyxl.load_workbook(file_name, read_only=False)
        writer = pandas.ExcelWriter(file_name, engine='openpyxl')
        writer.book = excel
        writer.sheets = dict((ws.title, ws) for ws in excel.worksheets)
        features.to_excel(writer, sheet_name=str(multiplier), index=False)
        writer.save()
writer.close()
















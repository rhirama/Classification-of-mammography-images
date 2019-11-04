import cv2
import glob
import os
import openpyxl
import xlsxwriter
import pandas
import numpy as np
import img_loader as imlo

imgs_54BND = 'D:/Users/Rodrigo S. Hirama/Documentos/EACH/IC/Classification-of-mammography-images/Python/' \
             'Contours54BND/*.jpg'
imgs_57EDG = 'D:/Users/Rodrigo S. Hirama/Documentos/EACH/IC/Classification-of-mammography-images/Python/' \
             'Contours57EDG/*.jpg'

_destination = 'D:/Users/Rodrigo S. Hirama/Documentos/EACH/IC/poly_models'

_paths = [imgs_54BND, imgs_57EDG]
writer = None
features = None


def get_poly_models(paths, ini_mult=0.001, end_mult=0.01, step=0.001, n_polymodels=False):
    multipliers = [ini_mult]
    if n_polymodels:
        multipliers = np.arange(ini_mult, end_mult, step)
    for img_path in paths:
        for m in multipliers:
            for name in sorted(glob.glob(img_path)):
                img_color, img_gray = imlo.load_img(name)
                contours = imlo.pre_process(img_gray)
                cnt = imlo.max_area_contour(contours)

                perimeter = int(cv2.arcLength(cnt, True))
                epsilon = m * perimeter
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                approx = np.squeeze(approx)
                poly_shape = approx.shape[0]
                approx = approx.reshape((poly_shape*2))

                name = os.path.splitext(os.path.basename(name))[0]
                write_file(_destination, folder=str(m), file=name, poly_p=approx)


def write_file(destination, folder, file, poly_p, extension='.txt'):
    if not os.path.exists(destination):
        print("caminho de destino nao existe :(")
        return
    os.chdir(destination)
    if not os.path.exists(folder):
        os.mkdir(folder)
    os.chdir(folder)
    np.savetxt(file+extension, poly_p, fmt='%d')


get_poly_models(_paths)




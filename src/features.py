import cv2
import importlib
import glob
import os
import numpy as np
from pandas import DataFrame
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
comp_mod = importlib.import_module('Compactness')
si_mod = importlib.import_module('Spiculation_old')
ff_mod = importlib.import_module('FourierDescriptor')
fd_mod = importlib.import_module('Fractal')


# imgs_54BND = '/home/who/Documents/cg/Classification-of-mammography-images/Imagens/Contours54BND/*.jpg'
# imgs_57EDG = '/home/who/Documents/cg/Classification-of-mammography-images/Imagens/Contours57EDG/*.jpg'
imgs_54BND = 'Contours54BND/*.jpg'
imgs_57EDG = 'Contours57EDG/*.jpg'
diagnosis57 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
diagnosis54 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

files_paths = [(imgs_54BND, 54, diagnosis54), (imgs_57EDG, 57, diagnosis57)]
multipliers = np.arange(0.011, 0.02, 0.0005)
labels = ['name', 'compactness', 'si', 'fourier_factor', 'fd2Dbox', 'fd1Dbox', 'fd2Druler', 'fd1Druler']

for img_path, dataset, diag in files_paths:
    for m in multipliers:
        m = round(m, 4)
        features = []
        row = []
        row_labels = []
        id_img = 0
        for name in sorted(glob.glob(img_path)):
            print(name)
            img_color, img_gray = img_loader_mod.load_img(name)
            canvas = img_loader_mod.create_clear_canvas(img_gray)
            contours = img_loader_mod.pre_process(img_gray)
            cnt = max_area_contour(contours)
            row_labels.append(str(id_img))

            perimeter = int(cv2.arcLength(cnt, True))
            epsilon = m * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(canvas, [approx], 0, (255, 255, 255), 1)
            contours, hierarchy = cv2.findContours(canvas, cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_NONE)  # find contours of canvas
            cnt = max_area_contour(contours)

            comp = comp_mod.compactness(approx)
            si = si_mod.calculate_si(approx)
            fd2dbc = fd_mod.fractal_dimension_boxcount(canvas)
            fd2drm = fd_mod.ruler_fractal_dimension(cnt)

            (N, _, n) = approx.shape
            np.squeeze(approx)
            Zn = cv2.ximgproc.fourierDescriptor(approx)
            Z0 = ff_mod.normalizeFourierDescriptors(Zn)

            ff = ff_mod.fourierFactor(Z0)[0][0]

            # make 1d image and save it on a temporary png
            cnt = np.squeeze(approx)
            one_d = img_loader_mod.make_1d_contour(cnt)
            fig = plt.figure(frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.plot(one_d, linewidth=1)
            fig.savefig('temp')
            plt.close(fig)

            _, plot_gray = img_loader_mod.load_img("temp.png")
            img_gray = cv2.bitwise_not(plot_gray)  # invert colors on the temp png
            ret, thresh = cv2.threshold(plot_gray, 20, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = max_area_contour(contours)

            fd1dbc = fd_mod.fractal_dimension_boxcount(thresh)
            fd1drm = fd_mod.ruler_fractal_dimension(cnt)

            row.append(os.path.basename(name))
            row.append(comp)
            row.append(si)
            row.append(ff)
            row.append(fd2dbc)
            row.append(fd1dbc)
            row.append(fd2drm)
            row.append(fd1drm)
            features.append(row)
            row = []
            id_img += 1
        df = DataFrame(features, index=row_labels, columns=labels)
        df.insert(1, 'diagnosis', diag, allow_duplicates=True)
        df.to_csv(str(dataset)+'_'+str(m)+'.csv')

import cv2
import numpy as np
import glob
import pandas as pd
from matplotlib import pyplot as plt
import os

import src.img_loader as img_loader
import src.polygon as polygon
import src.segments as segments
import src.fourierDescriptor as fourierDescriptor
import src.fractal as fractal
from src.compactness import compactness
imgs_57EDG = 'images/Contours57EDG/*.jpg'
imgs_54BND = 'images/Contours54BND/*.jpg'

diagnosis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
             0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1]

paths = [imgs_57EDG, imgs_54BND]
eps_multipliers = np.linspace(0.001, 0.02, 20)
cont = 0

if __name__ == "__main__":
    for mult in eps_multipliers:
        mult = round(mult, 3)
        opencv_feats = {
            "img": [],
            "diag": [],
            "c": [],
            "si": [],
            "fcc": [],
            "ff": [],
            "fdbc2d": [],
            "fdbc1d": [],
            "fdrm2d": [],
            "fdrm1d": []
        }
        pb_feats = {
            "img": [],
            "diag": [],
            "c": [],
            "si": [],
            "fcc": [],
            "ff": [],
            "fdbc2d": [],
            "fdbc1d": [],
            "fdrm2d": [],
            "fdrm1d": []
        }

        for imgs_path in paths:
            for img_name in sorted(glob.glob(imgs_path)):
                img_color, img_gray = img_loader.load_img(img_name)
                contours = img_loader.pre_process(img_gray)

                poly = img_loader.max_area_contour(contours[0])
                poly = np.squeeze(poly)

                perimeter = cv2.arcLength(poly, True)
                epsilon = round(mult * perimeter)

                polydp = polygon.poly_model(poly, epsilon)
                dp_per = cv2.arcLength(polydp, True)

                polypb = polygon.reduce_poly(poly, polydp.shape[0])
                polypb = polypb.reshape((polypb.shape[0], 1, polypb.shape[1]))
                pb_per = cv2.arcLength(polypb, True)

                models = [(polydp, dp_per), (polypb, pb_per)]
                i = 0
                for model, per in models:
                    feat_row = {}
                    img = img_loader.create_clear_canvas(img_gray) * 255
                    cv2.fillPoly(img, [model], (0, 0, 0))

                    comp = compactness(model)
                    segs = segments.get_segments(model)
                    spics, concave_len = segments.get_spicules(segs, img, cont)
                    si = segments.spiculation_index(spics)
                    fcc = segments.fractional_concavity(concave_len, per)

                    Zn = cv2.ximgproc.fourierDescriptor(model)
                    Z0 = fourierDescriptor.normalizeFourierDescriptors(Zn)
                    ff = fourierDescriptor.fourierFactor(Z0)[0][0]

                    canvas = np.zeros(img_gray.shape, np.uint8)
                    cv2.drawContours(canvas, [model], 0, (255, 255, 255), 1)
                    fractal_contour = cv2.findContours(canvas, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    fractal_contour = img_loader.max_area_contour(fractal_contour[0])

                    fdbc2d = fractal.fractal_dimension_boxcount(canvas)
                    fdrm2d = fractal.ruler_fractal_dimension(fractal_contour)

                    cnt = np.squeeze(model)
                    one_d = img_loader.make_1d_contour(cnt)

                    # make 1d image and save it on a temporary png
                    fig = plt.figure(frameon=False)
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.plot(one_d)
                    fig.savefig('temp')
                    plt.close(fig)

                    _, one_d_img = img_loader.load_img("temp.png")
                    one_d_img = cv2.bitwise_not(one_d_img)  # invert colors on the temp png
                    ret, thresh = cv2.threshold(one_d_img, 20, 255, cv2.THRESH_BINARY)

                    fdbc1d = fractal.fractal_dimension_boxcount(thresh)
                    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    cnt = img_loader.max_area_contour(contours[0])
                    fdrm1d = fractal.ruler_fractal_dimension(cnt)
                    img_name = os.path.basename(os.path.splitext(img_name)[0])

                    if i % 2 == 0:
                        opencv_feats["img"].append(img_name)
                        opencv_feats["diag"].append(diagnosis[cont % len(diagnosis)])
                        opencv_feats["c"].append(comp)
                        opencv_feats["si"].append(si)
                        opencv_feats["fcc"].append(fcc)
                        opencv_feats["ff"].append(ff)
                        opencv_feats["fdbc2d"].append(fdbc2d)
                        opencv_feats["fdbc1d"].append(fdbc1d)
                        opencv_feats["fdrm2d"].append(fdrm2d)
                        opencv_feats["fdrm1d"].append(fdrm1d)
                    else:
                        pb_feats["img"].append(img_name)
                        pb_feats["diag"].append(diagnosis[cont % len(diagnosis)])
                        pb_feats["c"].append(comp)
                        pb_feats["si"].append(si)
                        pb_feats["fcc"].append(fcc)
                        pb_feats["ff"].append(ff)
                        pb_feats["fdbc2d"].append(fdbc2d)
                        pb_feats["fdbc1d"].append(fdbc1d)
                        pb_feats["fdrm2d"].append(fdrm2d)
                        pb_feats["fdrm1d"].append(fdrm1d)
                    i += 1
                print(cont)
                cont += 1
        pd.DataFrame.from_dict(opencv_feats, orient="index").to_csv(str(mult)+"_opencv.csv")
        pd.DataFrame.from_dict(pb_feats, orient="index").to_csv(str(mult)+"_pb.csv")

    print("pintao")

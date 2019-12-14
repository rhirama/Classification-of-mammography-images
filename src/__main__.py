import cv2
import numpy as np
import glob
import pandas as pd
from matplotlib import pyplot as plt
import os

import img_loader
import polygon
import segments
import fourierDescriptor
import fractal
from compactness import compactness
imgs_57EDG = 'images/Contours57EDG/*.jpg'
imgs_54BND = 'images/Contours54BND/*.jpg'

paths = [imgs_57EDG, imgs_54BND]
eps_multipliers = np.linspace(0.001, 0.02, 20)
cont = 0

if __name__ == "__main__":
    for mult in eps_multipliers:
        opencv_feats = {}
        pb_feats = {}
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

                    feat_row["img"] = os.path.basename(os.path.splitext(img_name)[0])
                    feat_row["c"] = comp
                    feat_row["si"] = si
                    feat_row["fcc"] = fcc
                    feat_row["ff"] = ff
                    feat_row["fdbc2d"] = fdbc2d
                    feat_row["fdbc1d"] = fdbc1d
                    feat_row["fdrm2d"] = fdrm2d
                    feat_row["fdrm1d"] = fdrm1d
                    if i % 2 == 0:
                        opencv_feats[cont] = feat_row
                    else:
                        pb_feats[cont] = feat_row
                    i += 1

                print(cont)
                cont += 1
        pd.DataFrame.from_dict(opencv_feats, orient="index").to_csv(str(mult)+"_opencv.csv")
        pd.DataFrame.from_dict(pb_feats, orient="index").to_csv(str(mult)+"_pb.csv")

    print("pintao")

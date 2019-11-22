import cv2
import numpy as np

import img_loader

if __name__ == "__main__":
    img_name = "Contours54BND/CIRC_DB_244.jpg"
    img_color, img_gray = img_loader.load_img(img_name)
    canvas = img_loader.create_clear_canvas(img_gray)
    contours = img_loader.pre_process(img_gray)
    print(contours[0])
    cnt = img_loader.max_area_contour(contours[0])

    print(type(cnt))
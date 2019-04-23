import cv2
import numpy as np

from Fractal import dist


def load_img(name):
    img = cv2.imread(name)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray


def create_clear_canvas(img):
    canvas = np.zeros(img.shape, np.uint8)
    return canvas


def pre_process(img_gray):
    # dependendo da imagem modificar os valores de thresh e maxval, para as imagens do Ricardo 20, 255
    ret, thresh = cv2.threshold(img_gray, 20, 255,
                                cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def make_1d_contour(contour):
    # get Moment of contour: https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    M = cv2.moments(contour)
    c_x = int(M['m10'] / M['m00'])
    c_y = int(M['m01'] / M['m00'])
    centroid = [c_x, c_y]
    # x, y, w, h = cv2.boundingRect(np.array(contour))
    # centroid = [x, y]
    return np.array([[dist(contour[i], centroid)] for i in range(len(contour))])

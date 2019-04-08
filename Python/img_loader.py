import cv2
import numpy as np


def load_img(name):
	img = cv2.imread(name)
	img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img, img_gray


def create_clear_canvas(img):
	canvas = np.zeros(img.shape, np.uint8)
	return canvas


def pre_process(img_gray):
	ret, thresh = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)  # dependendo da imagem modificar os valores de thresh e maxval, para as imagens do Ricardo 20, 255
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours


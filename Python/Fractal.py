import numpy as np
import scipy
import cv2
import random
from math import sqrt, inf
from matplotlib import pyplot as plt


def fractal_dimension_boxcount(z, threshold=120):
    # Only for 2d image
    assert (len(z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(z, k):
        s = np.add.reduceat(
            np.add.reduceat(z, np.arange(0, z.shape[0], k), axis=0),
            np.arange(0, z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((s > 0))[0])

    # Transform Z into a binary array
    z = (z > threshold)

    # Minimal dimension of image
    p = min(z.shape)

    # Greatest power of 2 less than or equal to p
    # Extract the exponent
    n = int(np.floor(np.log(p) / np.log(2)))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    plt.plot(np.log(counts), np.log(sizes))
    plt.show()
    return -coeffs[0]


def ruler_fractal_dimension(contour):
    # https://stackoverflow.com/questions/37041008/python-boundingrect-with-list-of-points
    contour = np.squeeze(contour)  # removes unwanted empty dimensions from array
    contour = normalize_contour(contour)

    steps = []
    perimeters = []

    step_len = 0.3
    # step_len = 0.050
    for i in range(0, 7):
        perimeter = 0
        step_sizes = 0
        for j in range(0, 50):
            new_perim, actual_step_size = ruler_method(contour, step_len)
            if new_perim == -1:
                break
            perimeter += new_perim
            step_sizes += actual_step_size
        if perimeter <= 0:
            break

        print('step_len: ', step_sizes / 50, ' perimeter: ', perimeter / 50)
        steps.append(step_sizes / 50)
        perimeters.append(perimeter / 50)

        step_len = step_len / 2
        # step_len += 0.025

    coeffs = np.polyfit(np.log(steps), np.log(perimeters), 1)
    # print(1-coeffs[0])
    # plt.plot(np.log(steps), np.log(perimeters))
    # plt.plot(np.log(steps), np.log(steps)*coeffs[0]+coeffs[1])
    # plt.show()
    print(1 - coeffs[0])
    return 1 - coeffs[0]


def ruler_method(contour, step_len):
    contour = shuffle_contour(contour)
    start = contour[0]
    perimeter = 0
    i = 0
    steps = 0
    while i < len(contour):
        distance = dist(start, contour[i])
        if distance > step_len:
            distance2 = dist(start, contour[i - 1])
            if abs(distance - step_len) < abs(distance2 - step_len):  # current point is closest to step_len
                perimeter += distance
                start = contour[i]
                i += 1  # start looking at next point
            else:  # last point was closest to step_len, evaluate current point again
                perimeter += distance2
                # if step_len is smaller than the distance between start and the next point the walk wont proceed
                if np.all(start - contour[i - 1] == 0):
                    print('step_len is too small')
                    return -1, -1
                start = contour[i - 1]
            steps += 1
            continue  # skip the "i += 1" line
        i += 1
    perimeter += dist(start, contour[i - 1])  # adds the unfinished line to perimeter

    return perimeter, perimeter/steps


def dist(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def shuffle_contour(contour):
    start = random.randrange(0, len(contour))
    begin = contour[0: start - 1]
    end = contour[start - 1: len(contour)]
    contour = np.vstack((end, begin))
    return contour


def normalize_contour(contour):
    new_canvas = np.zeros(contour.shape)  # canvas that stores the float result, because contour's dtype is uint8
    max_x = 0
    max_y = 0
    min_x = inf
    min_y = inf

    for pixel in contour:
        if pixel[0] > max_x:
            max_x = pixel[0]
        if pixel[1] > max_y:
            max_y = pixel[1]
        if pixel[0] < min_x:
            min_x = pixel[0]
        if pixel[1] < min_y:
            min_y = pixel[1]

    x_amplitude = max_x - min_x
    y_amplitude = max_y - min_y
    # print("x_amplitude:", x_amplitude, "y_amplitude:", y_amplitude)
    amplitude = float(max(x_amplitude, y_amplitude))

    for i in range(len(contour)):
        # print("pixel[0]:", contour[i][0], "pixel[1]:", contour[i][1])
        new_x = (contour[i][0] - min_x) / amplitude
        new_y = (contour[i][1] - min_y) / amplitude
        new_canvas[i] = [new_x, new_y]

    return new_canvas


# # box-counting test
# image = cv2.imread('sier_9_1000.jpg')
# img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(type(img_gray))
# thresh= cv2.resize(thresh, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
# thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)

# ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
# canny = cv2.Canny(thresh, 100, 150)
# print("Minkowski–Bouligand dimension (computed): ", fractal_dimension(thresh))

# cv2.imshow('zika', thresh)
# cv2.waitKey(0)


# # ruler-method test
# def max_area_contour(contours):
#     cnt = contours[0]
#     max_area = cv2.contourArea(cnt)
#
#     for cont in contours:
#         if cv2.contourArea(cont) > max_area:
#             cnt = cont
#             max_area = cv2.contourArea(cont)
#     return cnt
#
#
# img_color = cv2.imread('tametwindragon.jpg')
# img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
# canvas = np.zeros(img_gray.shape, np.uint8)
# ret, thresh = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
# # RETR_EXTERNAL for getting only the outer contour and CHAIN_APPROX_NONE to return a list of contour points
# im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# cnt = max_area_contour(contours)
#
# # make polygon approx.
# perimeter = int(cv2.arcLength(cnt, True))
# epsilon = 0.001 * perimeter
# approx = cv2.approxPolyDP(cnt, epsilon,
#                           True)  # parâmetros para testar: epsilon(dita o quao simplificada fica a figura)
# cv2.drawContours(canvas, [approx], 0, (255, 255, 255), 1) # write polygon approx on canvas
# im2, contours, hierarchy = cv2.findContours(canvas, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # find contours of canvas
# cnt = max_area_contour(contours)
#
# # # show contour for debugging
# # cv2.drawContours(img_color, cnt, -1, (255, 0, 255))
# # cv2.namedWindow('help', cv2.WINDOW_NORMAL)
# # cv2.resizeWindow('help', 1200, 1200)
# # cv2.imshow('help', img_color)
# # cv2.waitKey(0)
# # # exit(0)
#
# random.seed(4)
# ruler_fractal_dimension(cnt)

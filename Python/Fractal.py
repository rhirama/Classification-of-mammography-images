import numpy as np
import scipy
import cv2
import random
from math import sqrt
from matplotlib import pyplot as plt


def fractal_dimension(z, threshold=120):
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
    _, _, w, h = cv2.boundingRect(np.array(contour))
    diag = sqrt(w**2 + h**2)
    feret_diam = min(w, h)

    steps = []
    perimeters = []
    
    step_len = 2
    for i in range(0, 12):
        # step_len = diag*1/(2*2**(1/2*i))
        if step_len > feret_diam / 3:
            continue

        perimeter = 0
        for j in range(0, 50):
            perimeter += ruler_method(contour, step_len)
        if perimeter < 0:
            break

        print('step_len: ', step_len, ' perimeter: ', perimeter/50)
        steps.append(step_len)
        perimeters.append(perimeter/50)
        
        step_len = step_len*2

    coeffs = np.polyfit(np.log(steps), np.log(perimeters), 1)
    # print(1-coeffs[0])
    # plt.plot(np.log(steps), np.log(perimeters))
    # plt.plot(np.log(steps), np.log(steps)*coeffs[0]+coeffs[1])
    # plt.show()
    return 1-coeffs[0]


def ruler_method(contour, step_len):
    contour = shuffle_contour(contour)
    start = contour[0]
    perimeter = 0
    i = 0
    while i < len(contour):
        distance = dist(start, contour[i])
        if distance > step_len:
            distance2 = dist(start, contour[i-1])
            if abs(distance - step_len) < abs(distance2 - step_len):  # current point is closest to step_len
                perimeter += distance
                start = contour[i]
                i += 1  # start looking at next point
            else:  # last point was closest to step_len, evaluate current point again
                perimeter += distance2
                # if step_len is smaller than the distance between start and the next point the walk wont proceed
                if np.all(start - contour[i-1] == 0):
                    print('step_len is too small')
                    return -1
                start = contour[i-1]
            continue
        i += 1
    perimeter += dist(start, contour[i-1])  # adds the unfinished line to perimeter

    return perimeter


def dist(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def shuffle_contour(contour):
    start = random.randrange(0, len(contour))
    begin = contour[0: start-1]
    end = contour[start-1: len(contour)]
    contour = np.vstack((end, begin))
    return contour


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
# image = cv2.imread('tametwindragon.jpg')
# img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # img_gray = cv2.bitwise_not(img_gray)  # invert colors on the temp png
# ret, thresh = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
# # RETR_EXTERNAL for getting only the outer contour and CHAIN_APPROX_NONE to return a list of contour points
# im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cnt = max_area_contour(contours)
# 
# # # show contour
# # cv2.drawContours(im2, contours, -1, (255,0,255))
# # cv2.namedWindow('help',cv2.WINDOW_NORMAL)
# # cv2.resizeWindow('help', 1200,1200)
# # cv2.imshow('help', im2)
# # cv2.waitKey(0)
# # exit(0)
# 
# random.seed(4)
# print(ruler_fractal_dimension(cnt))

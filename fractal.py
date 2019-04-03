import numpy as np
import scipy
import cv2
from matplotlib import pyplot as plt


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def fractal_dimension(z, threshold=120):
    # Only for 2d image
    assert (len(z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(z, k):
        s = np.add.reduceat(
            np.add.reduceat(z, np.arange(0, z.shape[0], k), axis=0),
            np.arange(0, z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((s > 0) & (s < k * k))[0])

    # Transform Z into a binary array
    z = (z > threshold)

    # Minimal dimension of image
    p = min(z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

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


image = cv2.imread("Sierpinski.jpg")
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(type(img_gray))

thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 3, 2)
thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 3, 2)
print(thresh.shape)

cv2.imshow('zika', thresh)
cv2.waitKey(0)
print(thresh)
print("Minkowski–Bouligand dimension (computed): ", fractal_dimension(thresh))

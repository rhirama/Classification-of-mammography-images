import numpy as np
import scipy
import cv2
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


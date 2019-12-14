import cv2
import numpy as np
from numpy.linalg import norm
from math import acos


def poly_model(poly, epsilon):
    cnt = cv2.approxPolyDP(poly, epsilon, closed=True)
    return cnt


def reduce_poly(pl, num):
    poly = pl
    numv = poly.shape[0]
    imp = np.zeros(numv)

    for v in range(0, imp.shape[0], 1):
        imp[v] = vertex_importance(v, poly, numv)

    while numv > num:
        i = np.argmin(imp)

        if i < numv:
            if np.array_equal(poly[(i - 1) % poly.shape[0]], poly[(i + 1) % poly.shape[0]]):
                poly = np.delete(poly, i - 1, 0)
                imp = np.delete(imp, i - 1)
                numv = numv - 1

            poly = np.delete(poly, i, 0)
            imp = np.delete(imp, i)
            vp = i % poly.shape[0]

        else:
            vp = 0
        numv = numv - 1

        vm = (i - 1) % numv

        imp[vp] = vertex_importance(vp, poly, numv)
        imp[vm] = vertex_importance(vm, poly, numv)

    return poly


def vertex_importance(v, poly, numv):
    vp = (v + 1) % numv
    vm = (v - 1) % numv

    dir1 = poly[v] - poly[vm]
    dir2 = poly[vp] - poly[v]
    len1 = norm(dir1)
    len2 = norm(dir2)

    len1len2 = len1 * len2
    return abs(acos(np.dot(dir1, dir2) / round(len1len2))) * len1len2

import numpy as np
import math


def angle(p1, p2, p3):  # calcula o angulo entre dois vetores usando o produto escalar

    ang = 0
    x1, y1 = p1[0] - p2[0], p1[1] - p2[1]
    x2, y2 = p3[0] - p2[0], p3[1] - p2[1]

    dist1 = math.sqrt(x1**2 + y1**2)
    dist2 = math.sqrt(x2**2 + y2**2)

    numer = (x1 * x2 + y1 * y2)
    denom = dist1 * dist2
    ang_aux = math.degrees(math.acos(numer/denom))

    z_direction = ((p2[0] - p1[0])*(p3[1] - p2[1])) - ((p3[0] - p2[0])*(p2[1] - p1[1]))

    if z_direction > 0:
        ang = 360 - ang_aux
    
    elif z_direction < 0:
        ang = ang_aux

    return ang, dist2


def weighting(angles):
    angs = 0
    div = 1
    ang_thresh = sum(angles) / angles.__len__()
    for i in angles:
        if i < ang_thresh:
            div += 1
            angs += i

    ang_weight = angs / div
    return ang_weight


def calculate_si(approx, perimeter):
    numer = 0
    denom = 1
    last_180 = 0
    length = 0

    angles = []
    sizes = []
    aux_ang = []

    for i in range(len(approx)):
        line1 = approx[i - 2, 0]
        ref = approx[i - 1, 0]
        line2 = approx[i, 0]

        ang, v_length = angle(line1, ref, line2)

        angles.append(ang)
        sizes.append(v_length)

        if ang > 180:
            last_180 = i

    var = angles.__len__() - last_180

    for i in range(last_180 - angles.__len__() + 1, angles.__len__() - var):

        if angles[i] <= 180:
            length += sizes[i]
            aux_ang.append(angles[i])
        else:
            if aux_ang.__len__() < 1:
                continue
            ang = np.cos(np.degrees(weighting(aux_ang))) + 1
            numer += ang * length
            denom += length

            length = 0

    spiculation_index = numer/denom
    return spiculation_index

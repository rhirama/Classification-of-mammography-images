import math


def angle(p1, p2, p3):  # calcula o angulo entre dois vetores usando o produto escalar

    ang = 0
    x1, y1 = p1[0] - p2[0], p1[1] - p2[1]
    x2, y2 = p3[0] - p2[0], p3[1] - p2[1]

    dist1 = math.sqrt(x1 ** 2 + y1 ** 2)
    dist2 = math.sqrt(x2 ** 2 + y2 ** 2)

    numer = (x1 * x2 + y1 * y2)
    denom = dist1 * dist2
    ang_aux = math.degrees(math.acos(round(numer / denom, 4)))

    z_direction = ((p2[0] - p1[0]) * (p3[1] - p2[1])) - ((p3[0] - p2[0]) * (p2[1] - p1[1]))

    if z_direction > 0:
        ang = 360 - ang_aux

    elif z_direction < 0:
        ang = ang_aux

    return ang, dist2


def calculate_si(approx):
    numer = 0
    denom = 0

    for i in range(len(approx)):
        line1 = approx[i - 2, 0]
        ref = approx[i - 1, 0]
        line2 = approx[i, 0]

        ang, v_length = angle(line1, ref, line2)

        numer += (1 + math.cos(math.radians(ang))) * v_length
        denom += v_length

    spiculation_index = numer/denom
    return spiculation_index


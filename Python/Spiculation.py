import numpy as np
import math

def angle(x1, y1, x2, y2):  # calcula o angulo entre dois vetores usando o produto escalar
    dist1 = math.sqrt(x1**2 + y1**2)
    dist2 = math.sqrt(x2**2 + y2**2)

    numer = (x1 * x2 + y1 * y2)
    denom = dist1 * dist2
    return math.degrees(math.acos(numer/denom)), dist2

def calcultateSpiculationIndex(approx):
    numer = 0
    denom = 0

    for i in range(len(approx)):
        line2 = approx[i][0]
        ref = approx[i-1][0]
        line1 = approx[i-2][0]
        x1, y1 = line1[0] - ref[0], line1[1] - ref[1]
        x2, y2 = line2[0] - ref[0], line2[1] - ref[1]

        ang, vLength = angle(x1, y1, x2, y2)

        #print(ang)

        numer += (1 + math.cos(math.radians(ang))) * vLength
        denom += vLength

    spiculationIndex = numer/denom
    return spiculationIndex

import cv2
import numpy as np
from math import sqrt
from math import cos


class Segment:
    def __init__(self, segment_id, x_initial, y_initial, x_final, y_final, x_mid_point, y_mid_point, length):
        self.segment_id = segment_id
        self.x_initial = x_initial
        self.y_initial = y_initial
        self.x_final = x_final
        self.y_final = y_final
        self.x_mid_point = x_mid_point
        self.y_mid_point = y_mid_point
        self.length = length


def get_angle(p0, p1, p2):
    """ compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    """

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return angle  # retorna o anglo em radianos


def get_segments(poly):
    # cria uma lista de pontos com o ponto médio de cada segmento
    i = 0
    segments = []
    total_length_mass = 0
    while i < len(poly) - 1:
        segment_id = i
        x_initial = poly[i, 0, 0]
        y_initial = poly[i, 0, 1]
        x_final = poly[(i + 1), 0, 0]
        y_final = poly[(i + 1), 0, 1]
        x_mid_point = int((x_initial + x_final) / 2)
        y_mid_point = int((y_initial + y_final) / 2)
        length = int(sqrt((x_final - x_initial) ** 2 + (y_final - y_initial) ** 2))
        total_length_mass += length
        segments.append(Segment(segment_id, x_initial, y_initial, x_final, y_final, x_mid_point, y_mid_point, length))
        i += 1

    # o primeiro e o último ponto da lista de pontos médios são o mesmo ponto. Necessário para poder percorrer o ciclo
    # inteiro de pontos
    segments.append(segments[0])
    return segments


def get_spicules(segments, img, cont):
    n_spicule = 0
    total_length_concave = 0
    i = 0
    spicules = []
    while i < len(segments) - 1:
        # x e y são as coordenadas do ponto médio do segmento que une dois pontos médios pertencentes aos segmentos do nódulo
        x = int((segments[i].x_mid_point + segments[i + 1].x_mid_point) / 2)
        y = int((segments[i].y_mid_point + segments[i + 1].y_mid_point) / 2)
        if img[y, x] == 255:  # se o ponto médio encontrado for branco, significa que está fora do nódulo
            total_length_concave += (segments[i].length + segments[(i + 1) % len(segments)].length) / 2  # talvez tenha que ser /2 length?
            i += 1
        else:
            spicule = [segments[i], segments[(i + 1) % len(segments)]]
            j = i + 1
            while j < len(segments) - 1:
                x_last = int((segments[j].x_mid_point + segments[j + 1].x_mid_point) / 2)
                y_last = int((segments[j].y_mid_point + segments[j + 1].y_mid_point) / 2)

                if img[y_last, x_last] == 255:
                    break
                j += 1
                if segments[j % len(segments)] not in spicule:
                    spicule.append(segments[j % len(segments)])
            spicules.append(spicule)
            i = j
            n_spicule += 1
    return spicules, total_length_concave


def fractional_concavity(concave_len, perimeter):
    return concave_len/perimeter


def spiculation_index(spicules):
    SI = 0
    total_length_spicules = 0
    # print('n_spicule: ', n_spicule, '\n=======')
    for items in spicules:
        index = 0
        all_angles = []
        while index < len(items) - 1:
            segment1 = items[index]
            segment2 = items[index + 1]
            # print('segment: ', segment1.segment_id, ', length: ', segment1.length)
            # print('segment: ', segment2.segment_id, ', length: ', segment2.length)
            p1 = [segment1.x_initial, segment1.y_initial]
            p2 = [segment1.x_final, segment1.y_final]
            p3 = [segment2.x_final, segment2.y_final]
            all_angles.append(get_angle(p1, p2, p3))
            # print('p1: ',  p1, ' p2: ', p2, ' p3: ', p3)
            # print('angle radians: ', get_angle(p1, p2, p3))
            # print('angle degree: ', np.degrees(get_angle(p1, p2, p3)))
            # print('angle degree positive: ', np.degrees(get_angle(p1, p2, p3))+360)
            # print('cosseno: ', cos(get_angle(p1, p2, p3)))
            index += 1
        # necessario percorrer novamente para pegar o tamanho total da espícula e o tamanho total de todas as espiculas
        length_spicule = 0
        i = 0
        for item in items:
            if i == 0 or i == len(items) - 1:
                length_spicule += item.length / 2
            i += 1

        total_length_spicules += length_spicule
        mean_all_angles = sum(all_angles) / len(all_angles)
        mean_used_angles = 0
        count = 0
        for angle in all_angles:
            if angle <= mean_all_angles:
                mean_used_angles += angle
                count += 1

        mean_used_angles = mean_used_angles / count
        SI += (1 + cos(mean_used_angles)) * length_spicule

    return SI / total_length_spicules
    # print('mean angles: ', mean_all_angles)
    # print('mean_used_angles: ', mean_used_angles)
    # print('length spicule: ', length_spicule)
    # print('cos: ', cos(mean_used_angles))
    # print('SI intermediario: ', SI)
    # print('=======\n')

    # print('total_length_spicules: ', total_length_spicules)
    # print('SI final: ', SI/total_length_spicules)
    # print('total_length_mass: ', total_length_mass)
    # print('total_length_concave: ', total_length_concave)
    # print('fractional concavity: ', total_length_concave/total_length_mass)
    # str( total_length_concave / total_length_mass))
    # cv2.imshow("foo",img)

import numpy as np
import cv2
import math
import importlib

# Essa função calcula os descritores de Fourier. Ela não está sendo utilizada no código, pois estou utilizando a função
# que já está implementada no OpenCV. Não usar essa função do jeito que está, pois ao invés de colocar os descritores nos
# locais corretos do vetor, eu coloquei um append, mas deveria ter colocado um Zk[k] = ..., fora isso, ela está funcionando
# corretamente

def myFourierDescriptor(zn):
    N = zn.size
    Zk = np.array([])

    for k in range(round(-N/2), round(N/2)):
        result = 0
        for n in range(0, N):
            result += zn[n]*np.exp(-1j*2*math.pi*n*k/N)
        Zk = np.append(Zk, (1/N)*result)
    return Zk


# Calcula os descritores de fourier normalizados
def normalizeFourierDescriptors(Zn):
    # N é a quantidade de descritores de Fourier em Zn
    (N, m, o) = Zn.shape

    Z0 = Zn.copy()

    # Não tem o -1 no segundo parâmetro do range, pq a função range é exclusiva
    for k in range(round(-N/2), round(N/2)):
        if k == 0:
            Z0[k] = 0
        else:
            Z0[k] = Zn[k]/abs(Zn[1])
    return Z0


def fourierFactor(Z0):
    # N é a quantidade de descritores de Fourier em Zn
    (N, m, o) = Z0.shape
    numerador = 0

    # Tem o +1 no segundo parâmetro do range, pq a função range é exclusiva
    for k in range(round(-N/2 + 1), round(N/2) + 1):
        if k != 0:
            numerador += abs(Z0[k])/abs(k)
    
    denominador = 0

    # Tem o +1 no segundo parâmetro do range, pq a função range é exclusiva
    for k in range(round(-N/2 + 1), round(N/2) + 1):
        denominador += abs(Z0[k])

    print('numerador: ' + str(numerador))

    print('denominador: ' + str(denominador))
    # calcula o fator de Fourier
    FF = 1 - (numerador/denominador)

    return FF


def max_area_contour(contours):
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        actual = cv2.contourArea(cont)
        if actual > max_area:
            cnt = cont
            max_area = actual
    return cnt


# img_loader_mod = importlib.import_module('img_loader')
# name = 'C62_97LCROI.jpg'
# multiplier = 0.001
# img_color, img_gray = img_loader_mod.load_img(name)
# canvas = img_loader_mod.create_clear_canvas(img_gray)
# contours = img_loader_mod.pre_process(img_gray)
# cnt = max_area_contour(contours)
#
# perimeter = int(cv2.arcLength(cnt, True))
# epsilon = multiplier * perimeter
# approx = cv2.approxPolyDP(cnt, epsilon, True)

# Array com os pontos do contorno. Estou utilizando apenas 3 pontos, mas o correto é ler do arquivo.

# N vai conter a quantidade de pontos no vetor
(N, _, n) = cnt.shape
# reshape necessário para poder utilizar o openCV
# contornoReshape = contornoOriginal.reshape(N, 1, 2)
# calcula os descritores de Fourier utilizando a função do openCV
Zn = cv2.ximgproc.fourierDescriptor(approx)
print(Zn)
# calcula dos descritores de Fourier normalizados
Z0 = normalizeFourierDescriptors(Zn)
# calcula o Fator de Fourier
print(fourierFactor(Z0))




#=========================================#
#pontos (x,y) = (12,11), (10,9), (1,2)
#zn = x(n) + j*y(n) para n = 0, 1,..., N-1
#As duas linhas abaixo é para calcular os descritores de fourier utilizando a função que eu criei
#zn = np.array([12 + 11j, 10 + 9j, 1 + 2j])
#Zk = myFourierDescriptor(zn)
import cv2
import importlib
import glob
import os
import xlsxwriter

imgLoaderMod = importlib.import_module('ImgLoader')
spIndexMod = importlib.import_module('Spiculation')
compMod = importlib.import_module('Compactness')


def maxAreaContour(contours):
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    return cnt


workbook = xlsxwriter.Workbook('Toy.xlsx')
worksheet = workbook.add_worksheet()

for multiplier in [0.05, 0.01, 0.001]:
    features = (['Nome', 'Compacidade', 'Índice de Espiculação'],)
    worksheet = workbook.add_worksheet(str(multiplier))
    row = 0
    col = 0
    for name in glob.glob("D:/Users/Rodrigo S. Hirama/Imagens/Contours54BND/*.jpg"):
        
        imgColor, imgGray = imgLoaderMod.loadImg(name)
        canvas = imgLoaderMod.createClearCanvas(imgColor)
        contours = imgLoaderMod.preProcess(imgGray)
        cnt = maxAreaContour(contours)
        
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(imgColor, center, radius, (0, 255, 0), 1)
        #cv2.imshow('circulo', imgColor)
        #cv2.waitKey(0)
        
        #perimeter = 2 * math.pi * radius 
        perimeter = int(cv2.arcLength(cnt, True))
        epsilon = multiplier * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)# parâmetros para testar: epsilon(dita o quao simplificada fica a figura)
        cv2.drawContours(canvas, [approx],  0, (0, 0, 255), 1)
        #cv2.imshow('modelo poligonal', canvas)
        #cv2.waitKey(0)
        
        compNorm = compMod.compactness(approx)
        si = spIndexMod.calcultateSpiculationIndex(approx)
        name = os.path.basename(name)
        features += ([name, compNorm, si],)
        #print(compNorm)
        #print(si)
    for name, comp_n, si in (features):
        worksheet.write(row, col, name)
        worksheet.write(row, col + 1, comp_n)
        worksheet.write(row, col + 2, si)
        row += 1
workbook.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
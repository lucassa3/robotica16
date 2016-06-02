#!/usr/bin/python
#-*- coding:utf-8 -*-

from __future__ import print_function

import cv2
import numpy as np
import math

video = cv2.VideoCapture('IMG_1926.MOV') 

while True:
    ret, frame = video.read() #ret avalia se condição de captura do frame = true, enquanto o frame indica a caputra do frame do video em si

    src = frame
    dst = cv2.Canny(src, 50, 200) # aplica o detector de bordas de Canny à imagem src
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR) # Converte a imagem para BGR para permitir desenho colorido
    

    if True: # HoughLinesP
        lines = cv2.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
 
        print("Used Probabilistic Rough Transform")
        print("The probabilistic hough transform returns the end points of the detected lines")
        a,b,c = lines.shape
        
        line_sizes = []
        for i in range(a):
            # Faz uma linha ligando o ponto inicial ao ponto final, com a cor vermelha (BGR)
            cv2.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 255, 0), 3)
            
            line_size = math.sqrt((lines[i][0][2] - lines[i][0][0])**2 + (lines[i][0][3] - lines[i][0][1])**2)
            line_sizes.append(line_size)
                    
        
        biggest_line = line_sizes.index(max(line_sizes))
        line_sizes[biggest_line] = 0
        second_biggest = line_sizes.index(max(line_sizes))

        cv2.line(cdst, (int(round((lines[biggest_line][0][0] + lines[second_biggest][0][0])/2)), int(round((lines[biggest_line][0][1] + lines[second_biggest][0][1])/2))), (int(round((lines[biggest_line][0][2] + lines[second_biggest][0][2])/2)), int(round((lines[biggest_line][0][3] + lines[second_biggest][0][3])/2))), (0, 0, 255), 3)
            

#    else:    # HoughLines
#        # Esperemos nao cair neste caso
#        lines = cv2.HoughLines(dst, 1, math.pi/180.0, 50, np.array([]), 0, 0)
#        a,b,c = lines.shape
#        for i in range(a):
#            rho = lines[i][0][0]
#            theta = lines[i][0][1]
#            a = math.cos(theta)
#            b = math.sin(theta)
#            x0, y0 = a*rho, b*rho
#            pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
#            pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
#            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3)
#        print("Used old vanilla Hough transform")
#        print("Returned points will be radius and angles")

    cv2.imshow("detected lines", cdst)
    cv2.waitKey(1)

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
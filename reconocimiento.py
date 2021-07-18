import cv2 as cv
import numpy as np

"""Además de usar la librería de Opencv, usamos parte de la documentación
que tienen disponible en GitHub para el entrenamiento de los algoritmos.
En este caso, cargamos la info que utiliza para identificar qué es un rostro
y qué NO es un rostro. INCREIBLE Opencv!!!!!"""
ruidos = cv.CascadeClassifier("D:\DOCUMENTOS\python_udemy\proyecto\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")

camara = cv.VideoCapture(0)

while True:
    _, captura = camara.read()
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    caras = ruidos.detectMultiScale(grises, 1.3, 5) #maneja porcentajes reduciendo la cara a cuadritos - puntos

    for (x, y, e1, e2) in caras:
        cv.rectangle(captura, (x, y), (x + e1, y + e2), (255, 255, 0), 2)

    cv.imshow("Resultado Rostro", captura)

    if cv.waitKey(1) == ord('q'):
        break

camara.release()
cv.destroyAllWindows()
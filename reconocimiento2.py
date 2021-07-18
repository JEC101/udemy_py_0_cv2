#Reconocimiento de rostros a partir de archivos de video

import cv2 as cv
import os
import imutils

modelo = "fotosElonMusk"
ruta1 = "D:/DOCUMENTOS/python_udemy/proyecto"
ruta_completa = ruta1 + '/' + modelo

if not os.path.exists(ruta_completa):
    os.makedirs(ruta_completa)

camara = cv.VideoCapture("D:/DOCUMENTOS/python_udemy/proyecto/reconocimientofacial1/ElonMusk.mp4")


ruidos = cv.CascadeClassifier("D:\DOCUMENTOS\python_udemy\proyecto\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")


# ID para cada imagen
id = 0

while True:
    respuesta, captura = camara.read()
    if respuesta == False:
        break

    # bajamos el tamaño de la imagen capturada para mejorar el rendimiento
    captura = imutils.resize(captura, width = 640)

    # pasamos a escala de grises para que mejore el rendimiento de los cálculos
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)

    #copio las propiedades de cada captura
    id_captura = captura.copy() 

    #maneja porcentajes reduciendo la cara a cuadritos - puntos
    caras = ruidos.detectMultiScale(grises, 1.3, 5) 

    for (x, y, e1, e2) in caras:
        cv.rectangle(captura, (x, y), (x + e1, y + e2), (255, 255, 0), 2)
        rostro_capturado = id_captura[y:y+e2, x:x+e1]  #va almaceando partes del rostro en la carpeta

        # se captura el rostro detectado en al img
        rostro_capturado = cv.resize(rostro_capturado, (160, 160), interpolation = cv.INTER_CUBIC)
        # lo guardo en la carpeta creada
        cv.imwrite(ruta_completa+"/imagen_{}.jpg".format(id), rostro_capturado)
        # sumo 1 al id para ir numerando las imgs que vengan a continuación y no sobrescribir
        id += 1


    cv.imshow("Resultado Rostro", captura)

    if id == 350:
        break

camara.release()
cv.destroyAllWindows()
# Capa que dará la salida a partir del entrenamiento anterior

import cv2 as cv
import os

data_ruta = "D:\DOCUMENTOS\python_udemy\proyecto\Data"
lista_data = os.listdir(data_ruta)

entrenamiento_modelo_1 = cv.face.EigenFaceRecognizer_create()
entrenamiento_modelo_1.read("D:/DOCUMENTOS/python_udemy/Neurona_Eigen_Recognizer.xml")
ruidos = cv.CascadeClassifier("D:\DOCUMENTOS\python_udemy\proyecto\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")

camara = cv.VideoCapture(0)
while True:
    _, captura = camara.read()
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    id_captura = grises.copy()
    
    #detectar caras evitando los ruidos
    caras = ruidos.detectMultiScale(grises, 1.3, 5)

    for (x, y, e1, e2) in caras:     
        rostro_capturado = id_captura[y:y+e2, x:x+e1]
        rostro_capturado = cv.resize(rostro_capturado, (160, 160), interpolation = cv.INTER_CUBIC)

        # Resultado es igual a la comparación del rostro capturado con la data del entrenamiento del modelo
        resultado = entrenamiento_modelo_1.predict(rostro_capturado)
        cv.putText(captura, f"{resultado}", (x,y-80), 1, 1.3, (150, 255, 00), 1, cv.LINE_AA)
        if resultado[1] < 7500:
            cv.putText(captura, f"{lista_data[resultado[0]]}", (x,y - 20), 2, 1.3, (150, 255, 00), 1, cv.LINE_AA)

            cv.rectangle(captura, (x, y), (x+e1, y+e2), (200, 230, 75), 2)
        else:
            cv.putText(captura, "No encontrado", (x, y-20), 2, 0.7, (255, 50, 50), 1, cv.LINE_AA)

            cv.rectangle(captura, (x, y), (x+e1, y+e2), (200, 230, 75), 2)

    cv.imshow("Resultados", captura)

    if (cv.waitKey(1) == ord('q')):
        break

camara.release()
cv.destroyAllWindows()
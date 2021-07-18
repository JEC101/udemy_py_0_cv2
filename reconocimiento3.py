#Capa oculta para entrenamietno"

import cv2 as cv
import os
import numpy as np
from time import time

data_ruta = "D:\DOCUMENTOS\python_udemy\proyecto\Data"
lista_data = os.listdir(data_ruta)

ids = []

rostros_data = []
id = 0

tiempo_inicial = time()

#bucles para cargar la info 
for fila in lista_data:
    ruta_completa = data_ruta + '/' + fila
    
    print("Iniciando lectura")

    for imagen in os.listdir(ruta_completa):

        print(f"Agregando {imagen} de {fila} al array Ids")

        ids.append(id)

        # Lo transformamos a escala de grises en la misma operacion de append al array
        rostros_data.append(cv.imread(ruta_completa + '/' + imagen, 0))

    id += 1

    tiempo_final_lectura = time()
    tiempo_total_lectura = tiempo_final_lectura - tiempo_inicial
    print(f"Tiempo de lectura: {tiempo_total_lectura}")

# acá comienza el entrenamiendo del modelo
entrenamiento_modelo_1 = cv.face.EigenFaceRecognizer_create()
print("Iniciando el entrenamiento. Espere...")
entrenamiento_modelo_1.train(rostros_data, np.array(ids))

tiempo_final_entrenamiento = time()
tiempo_total_entrenamiento = tiempo_final_entrenamiento - tiempo_final_lectura
print(f"Tiempo de entrenamiento: {tiempo_total_entrenamiento}")

entrenamiento_modelo_1.write("Neurona_Eigen_Recognizer.xml")
print("Entrenamiento concluido")
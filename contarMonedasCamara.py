from cv2 import cv2
import numpy as np

# ordenando el "espacio de trabajo" - se define donde se trabaja
def ordenar_puntos(puntos):
    n_puntos = np.concatenate(puntos[0], puntos[1], puntos[2], puntos[3]).tolist()
    y_order = sorted(n_puntos, key = lambda n_puntos: n_puntos[1])
    x1_order = y_order[0 : 2]
    x1_order = sorted(x1_order, key = lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key = lambda x2_order: x2_order[0])

    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

def alineamiento(imagen, ancho, alto):
    imagen_alineada = None

    grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    tipo_umbral, umbral = cv2.threshold(grises, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("Umbral", umbral)

    contorno, jerarquia = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contorno = sorted(contorno, key = cv2.contourArea, reverse = True)[0:1]
    for c in contorno:
        # bajamos la "frecuencia de curvas" para hacerlo más armónico y con menos ruido
        epsilon = 0.01 * cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, epsilon, True)

        if len(aprox) == 4: #4 porque son los 4 ptos definidos arriba (encuentra el circulo)
            puntos = ordenar_puntos(aprox) # llamamos la función de arriba con los 4 puntos encontrados
            

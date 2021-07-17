from cv2 import cv2
import numpy as np

# ordenando el "espacio de trabajo" - se define donde se trabaja
def ordenar_puntos(puntos):
    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    y_order = sorted(n_puntos, key = lambda n_puntos: n_puntos[1])
    x1_order = y_order[0 : 2]
    x1_order = sorted(x1_order, key = lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key = lambda x2_order: x2_order[0])

    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

def alineamiento(imagen, ancho, alto): #la funcion define el area de trabajo
    imagen_alineada = None

    grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    tipo_umbral, umbral = cv2.threshold(grises, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("Umbral", umbral)

    contorno = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contorno = sorted(contorno, key = cv2.contourArea, reverse = True)[0:1]
    for c in contorno:
        # bajamos la "frecuencia de curvas" para hacerlo más armónico y con menos ruido
        epsilon = 0.01 * cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, epsilon, True)

        if len(aprox) == 4: #4 porque son los 4 ptos definidos arriba (encuentra el circulo)
            puntos = ordenar_puntos(aprox) # llamamos la función de arriba con los 4 puntos encontrados
            punto_s1 = np.float32(puntos) #convierte los ptos en matrices de enteros
            puntos_s2 = np.float32([0,0], [ancho, 0], [0, alto], [ancho, alto]) #definimos los 4 extremos del area de trabajo con coordenadas
            M = cv2.getPerspectiveTransform(punto_s1, puntos_s2) #no se cambia de lugar aunque gire la camara
            imagen_alineada = cv2.warpPerspective(imagen, M, (ancho, alto))
    
    return imagen_alineada

captura_video = cv2.VideoCapture(0)

while True:
    tipo_camara, camara = captura_video.read()
    if tipo_camara == False:
        break
    imagen_A6 = alineamiento(camara, ancho = 677,  alto = 480)
    """El ancho y alto en la linea superior estan definidos en pixeles,
    a partir del tamano (cm) de una hoja A6, que define su relacion aspecto (ratio),
    y la definicion de la camara a utilizar. Es importante escalar para poder reconocer"""

    if imagen_A6 is not None:
        puntos = []
        imagen_gris = cv2.cvtColor(imagen_A6, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imagen_gris, (5,5), 1)
        _, umbral_2 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV) #invertimos los colores del fondo y de la hoja
        cv2.imshow("Umbral 2", umbral_2)

        contorno_2 = cv2.findContours(umbral_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        #dibujar los contornos:
        cv2.drawContours(imagen_A6, contorno_2, -1, (255, 0, 0), 2)
        
        # suma la cantidad de cada moneda
        suma1 = 0.0
        suma2 = 0.0

        for c_2 in contorno_2:
            area = cv2.contourArea(c_2)
            momentos = cv2.moments(c_2) #encontramos el centro de masa del objeto. lo usamos para agregar etiquetas a la img
            if (momentos["m00"] == 0): #si el momento está estático
                momentos["m00"] == 1.0 #si está estático agregamos 1

            x = int(momentos["m10"] / momentos["m00"])
            y = int(momentos["m01"] / momentos["m00"])

            if area < 9300 and area > 8000:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, "$ 0,20", (x, y), font, 0.75, (0, 255, 0), 2)
                suma1 += 0.2

            if area < 7800 and area > 6500:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, "$ 0,10", (x, y), font, 0.75, (0, 255, 0), 2)
                suma2 += 0.1

        total = suma2 + suma1
        print(f"El total es de ${total:.2f}")

        cv2.imshow("Imagen A6", imagen_A6)
        cv2.imshow("Camara", camara)

    if cv2.waitKey(1) == ord('q'):
        break

captura_video.release()
cv2.destroyAllWindows()

        

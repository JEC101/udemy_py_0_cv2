#3 JUST COLOR PICKER

import cv2

# apertura de la imagen a trabajar
imagen = cv2.imread('proyecto\moendas_contorno\imagen1.jpg')

# pasar la img a escala de grises para trabajarla
# se puede elegir la cantidad de colores y bits
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Separamos los objetos de su entorno para trabajar la img
#Umbralizacion simple
_, img_umbral = cv2.threshold(grises, 100, 255, cv2.THRESH_BINARY)
# econtrar los contornos de la imagen trabajada
contorno, jerarquia = cv2.findContours(img_umbral, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagen, contorno, -1, (251, 100, 50), 3)


#Muestra las imagenes
cv2.imshow('Imagen original', imagen)
#cv2.imshow('Imagen en grises', grises)
#cv2.imshow('Imagen umbral', img_umbral)

cv2.waitKey(0) #valor 0 para imagenes estáticas / 1 para videos o cámara encendida (facial recognition)
cv2.destroyAllWindows()


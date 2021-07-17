import cv2
import numpy as np

valor_gauss = 3
valor_kernel = 3

original = cv2.imread('proyecto\moendas_contorno\monedas.jpg')
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# blurear la img para bajar el nivel de ruidos de los contornos
desenfoque = cv2.GaussianBlur(gris, (valor_gauss, valor_gauss), 0)

# Canny -- para segunda eliminación de ruidos
bordes = cv2.Canny(desenfoque, 60, 100)

# seleccionamos los contornos que nos interesa preservar
kernel = np.ones((valor_kernel, valor_kernel), np.uint8)
cierre = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)

contorno, jerarquias = cv2.findContours(cierre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Monedas encontradas: {}".format(len(contorno)))

cv2.drawContours(original, contorno, -1, (0, 0, 255), 2)

# Mostrar resultados
cv2.imshow("Grises", gris)
cv2.imshow("Desenfocada", desenfoque)
cv2.imshow("Canny", bordes)
cv2.imshow("Cierre", cierre)

cv2.imshow("Resultado", original)
cv2.waitKey(0)


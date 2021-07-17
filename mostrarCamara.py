import cv2 as cv

captura_video = cv.VideoCapture(0)
if not captura_video.isOpened():
    print("No se encontró la cámara")
    exit()

while True:
    tipo_camara, camara = captura_video.read()
    grises = cv.cvtColor(camara, cv.COLOR_BGR2GRAY)

    cv.imshow("Video en vivo", grises) 
    if cv.waitKey(1) == ord("q"):
        break

captura_video.release()
cv.destroyAllWindows()
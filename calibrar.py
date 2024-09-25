import cv2
import numpy as np
import time

DICCIONARIO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
tablero = cv2.aruco.CharucoBoard((6, 8), 0.03, 0.02, DICCIONARIO)
tablero.setLegacyPattern(True) #Esto es porque uso un tablaeo de anteriores versiones
detector = cv2.aruco.CharucoDetector(tablero)

# Podemos imprimir creando nosotros la imagen o descargando de...
# https://calib.io/pages/camera-calibration-pattern-generator
#
#paraimprimir = tablero.generateImage((600, 800))
#cv2.imshow("Para Imprimir", paraimprimir)
#cv2.waitKey()
#cv2.imwrite("charuco.tiff", paraimprimir)
#exit()

CPS = 1
esquinas = []
marcadores = []
tiempo = 1.0 / CPS

cap = cv2.VideoCapture(0)
if cap.isOpened():
    wframe = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hframe = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    final = False
    n = 0
    antes = time.time()
    while not final:
        ret, frame = cap.read()
        if not ret:
            final = True
        else:
            if time.time()-antes > tiempo:
                bboxs, ids, _, _ = detector.detectBoard(frame)
                if ids is not None and ids.size>8:
                        antes = time.time()
                        cv2.aruco.drawDetectedCornersCharuco(frame, bboxs, ids)
                        esquinas.append(bboxs)
                        marcadores.append(ids)
                        n = n + 1
            cv2.putText(frame, str(n), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
            cv2.imshow("WEBCAM", frame)
            if cv2.waitKey(20) > 0:
                final = True
    cap.release()
    cv2.destroyAllWindows()
    if n == 0:
        print("No se han capturado imágenes para hacer la calibración")
    else:
        print("Espera mientras calculo los resultados de calibración de la cámara...")

        cameraMatrixInt = np.array([[ 1000,    0, hframe/2],
                                    [    0, 1000, wframe/2],
                                    [    0,    0,        1]])
        distCoeffsInt = np.zeros((5, 1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
        (ret, cameraMatrix, distCoeffs, rvec, tvec, stdInt, stdExt, errores) = cv2.aruco.calibrateCameraCharucoExtended(charucoCorners=esquinas,
                                                                                                                charucoIds=marcadores,
                                                                                                                board=tablero,
                                                                                                                imageSize=(hframe, wframe),
                                                                                                                cameraMatrix=cameraMatrixInt,
                                                                                                                distCoeffs=distCoeffsInt,
                                                                                                                flags=flags,
                                                                                                                criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        with open('camara.py', 'w') as fichero:
            fichero.write("import numpy as np\n")
            fichero.write("cameraMatrix = np.")
            fichero.write(repr(cameraMatrix))
            fichero.write("\ndistCoeffs = np.")
            fichero.write(repr(distCoeffs))
            fichero.close()
            print("Los resultados de calibración se han guardado en el fichero camara.py")
else:
    print("No se pudo abrir la cámara")
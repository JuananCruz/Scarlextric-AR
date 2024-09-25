import math
import mathutils
import numpy as np
import cv2
import camara2 as camaraMovil
import camara
import pygame
import speech_recognition as sr
import time
import sounddevice

class CarControl:
    def __init__(self, initial_pos_world, velocidad_coche, ar, ar_dir, matrix_size, vidas=5):
        self.rot_coche = 0.0
        self.pos_coche = np.array(initial_pos_world[:2])  # Solo las coordenadas x, y
        self.velocidad_coche = velocidad_coche
        self.initial_pos_world = initial_pos_world
        self.moving = False  # Para controlar el movimiento del coche
        self.vidas = vidas  # Inicializar vidas
        self.vidas_iniciales = vidas
        self.ar = ar
        self.ar_dir = ar_dir
        self.matrix_size = matrix_size
        self.lap_times = []  # Lista para almacenar los tiempos de las vueltas
        self.start_time = None  # Tiempo de inicio de la vuelta actual
        self.last_lap_time = None  # Tiempo de la última vuelta
        self.nivel_superado = False  # Nivel superado inicialmente en falso

        # Definir las coordenadas de la línea de meta
        radius = matrix_size // 3
        center_x = matrix_size // 2
        center_y = matrix_size // 2
        self.finish_line_start = (center_x + radius - 10, center_y)
        self.finish_line_end = (center_x + radius + 10, center_y)
        
        self.crossed_start_line = False  # Bandera para saber si ha cruzado la línea de meta
        self.cruzando_meta = False  # Bandera para detectar si el coche está cruzando la línea de meta

        # Inicializa las matrices de rotación y escala
        self.mat_rot_coche_x = mathutils.Matrix.Rotation(math.radians(90.0), 4, 'X')
        self.mat_sca_coche = mathutils.Matrix.Scale(0.025, 4)

        # Inicializa pygame y carga los sonidos
        pygame.mixer.init()
        self.sound_motor = pygame.mixer.Sound('src/sound/motor.mp3')
        self.sound_motor.set_volume(0.2)  # Ajusta el volumen 
        self.sound_boom = pygame.mixer.Sound('src/sound/explosion.mp3')
        self.sound_boom.set_volume(0.1)  # Ajusta el volumen

        # Inicializa el reconocimiento de voz
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def escuchar_comando(self):
        with self.microphone as source:
            print("Escuchando...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

        try:
            comando = self.recognizer.recognize_google(audio, language='es-ES')
            print(f"Comando escuchado: {comando}")
            self.procesar_comando(comando.lower())
            return comando.lower()
        except sr.UnknownValueError:
            print("No se pudo entender el audio")
        except sr.RequestError as e:
            print(f"No se pudo solicitar resultados del servicio de reconocimiento de voz; {e}")
        return None

    def procesar_comando(self, comando):
        if "empezar" in comando:
            self.moving = True
            if not pygame.mixer.get_busy():
                self.sound_motor.play(-1)  # Reproducir sonido del motor en loop
            self.start_time = time.time()  # Iniciar el temporizador
        elif "detener" in comando:
            self.moving = False
            pygame.mixer.stop()  # Detener todos los sonidos
        elif "acelerar" in comando:
            self.velocidad_coche += 0.0030
            print(f"Velocidad aumentada a {self.velocidad_coche}")
        elif "frenar" in comando:
            self.velocidad_coche = max(0, self.velocidad_coche - 0.0020)
            print(f"Velocidad reducida a {self.velocidad_coche}")
        elif "reiniciar" in comando:
            self.vidas = self.vidas_iniciales
            self.moving = True  # Permitir que el coche vuelva a moverse
            self.nivel_superado = False  # Reiniciar el estado del nivel superado
            if not pygame.mixer.get_busy():
                self.sound_motor.play(-1)  # Reproducir sonido del motor en loop
            print("Juego reiniciado. Vidas restauradas a", self.vidas)

    def detectarPose(self, frame, idMarcador, tam, camId):
        diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        detector = cv2.aruco.ArucoDetector(diccionario)
        bboxs, ids, rechazados = detector.detectMarkers(frame)
        if ids is not None:
            for i in range(len(ids)):
                if ids[i] == idMarcador:
                    objPoints = np.array([[-tam / 2.0, tam / 2.0, 0.0],
                                          [tam / 2.0, tam / 2.0, 0.0],
                                          [tam / 2.0, -tam / 2.0, 0.0],
                                          [-tam / 2.0, -tam / 2.0, 0.0]])
                    if camId == 0:
                        ret, rvec, tvec = cv2.solvePnP(objPoints, bboxs[i], camara.cameraMatrix, camara.distCoeffs)
                    else:
                        ret, rvec, tvec = cv2.solvePnP(objPoints, bboxs[i], camaraMovil.cameraMatrix, camaraMovil.distCoeffs)
                    if ret:
                        return (True, rvec, tvec)
        return (False, None, None)

    def verificarSuperficie(self, pos_coche, circuito, matrix_size):
        # Convertir la posición del coche a coordenadas de la matriz del circuito
        matrix_x = int((pos_coche[0] + 0.5) * matrix_size)
        matrix_y = int((pos_coche[1] + 0.5) * matrix_size)
        
        # Asegurarse de que las coordenadas estén dentro de los límites de la matriz
        if 0 <= matrix_x < matrix_size and 0 <= matrix_y < matrix_size:
            # Devolver el valor de la matriz del circuito en esa posición
            return circuito[matrix_y, matrix_x], matrix_x, matrix_y
        else:
            # Devolver un valor por defecto si la posición está fuera de los límites
            return -1, matrix_x, matrix_y

    def verificarMeta(self):
        # Verifica si el coche cruza la línea de meta
        matrix_x = int((self.pos_coche[0] + 0.5) * self.matrix_size)
        matrix_y = int((self.pos_coche[1] + 0.5) * self.matrix_size)
        
        cruzando_meta_actualmente = (self.finish_line_start[0] <= matrix_x <= self.finish_line_end[0] and
                                     self.finish_line_start[1] - 1 <= matrix_y <= self.finish_line_start[1] + 1)

        if cruzando_meta_actualmente and not self.cruzando_meta:
            # El coche está cruzando la línea de meta por primera vez
            self.cruzando_meta = True
            if self.start_time:
                lap_time = time.time() - self.start_time
                self.lap_times.append(lap_time)
                self.last_lap_time = lap_time
                self.start_time = time.time()  # Reiniciar el temporizador para la próxima vuelta
                print(f"Tiempo de vuelta: {lap_time:.2f} segundos")
                # Detener el coche y marcar el nivel como superado
                self.moving = False
                self.nivel_superado = True
                pygame.mixer.stop()  # Detener todos los sonidos
        elif not cruzando_meta_actualmente:
            # El coche no está cruzando la línea de meta, reiniciar la bandera
            self.cruzando_meta = False

    def actualizarPosicion(self, circuito, matrix_size):
        # Capturar el frame de la cámara ID 0
        ret, frame = self.ar_dir.read()
        if not ret:
            raise ValueError("No se puede leer el frame de la cámara ID 0")

        ret, rvec, tvec = self.detectarPose(frame, 6, 0.15, 0)
        if ret:
            rmat = cv2.Rodrigues(rvec)[0]
            angle = math.atan2(rmat[1, 0], rmat[0, 0])
            self.rot_coche = angle

        # Solo mover el coche si el comando de voz lo permite y el nivel no ha sido superado
        if self.moving and not self.nivel_superado:
            # Crear matriz de rotación 2D para la orientación del coche
            rot_matrix = np.array([
                [math.cos(self.rot_coche), -math.sin(self.rot_coche)],
                [math.sin(self.rot_coche), math.cos(self.rot_coche)]
            ])

            # Mover el coche hacia adelante en su sistema de coordenadas locales
            movement_vector = np.array([0, -self.velocidad_coche])  # Hacia adelante (sur) en el sistema de coordenadas del coche
            self.pos_coche += rot_matrix @ movement_vector

            # Verificar la superficie debajo del coche
            superficie, matrix_x, matrix_y = self.verificarSuperficie(self.pos_coche, circuito, matrix_size)

            if superficie == 0:
                # Reiniciar posición del coche a la posición inicial
                self.pos_coche = np.array(self.initial_pos_world[:2])
                self.rot_coche = 0.0  # Opcionalmente reiniciar la rotación
                pygame.mixer.stop()  # Detener el sonido si el coche se reinicia
                self.sound_boom.play()  # Reproducir el sonido de colisión

                # Reducir una vida y mostrar mensaje
                self.vidas -= 1
                if self.vidas > 0:
                    print(f"Vidas restantes: {self.vidas}")
                    # Reproducir sonido del motor si el coche sigue moviéndose después de perder una vida
                    if self.moving:
                        self.sound_motor.play(-1)  # Reproducir sonido del motor en loop
                else:
                    print("Has perdido todas las vidas")
                    self.moving = False  # Detener el coche si se pierden todas las vidas
            else:
                # Verificar si cruza la meta
                self.verificarMeta()

        mat_loc_coche = mathutils.Matrix.Translation((self.pos_coche[0], self.pos_coche[1], self.initial_pos_world[2]))
        mat_rot_coche_y = mathutils.Matrix.Rotation(self.rot_coche, 4, 'Y')
        meshpose_coche = mat_loc_coche @ self.mat_rot_coche_x @ mat_rot_coche_y @ self.mat_sca_coche

        return meshpose_coche
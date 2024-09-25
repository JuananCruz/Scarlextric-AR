import cv2
import numpy as np
import pyrender
import trimesh
import mathutils
import math
import cuia
import camara2 as camara
from control import CarControl
from datetime import datetime
import threading
import time
import sounddevice

def fromOpencvToPyrender(rvec, tvec):
    pose = np.eye(4)
    pose[0:3, 3] = tvec.T
    pose[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
    pose[[1, 2]] *= -1
    pose = np.linalg.inv(pose)
    return pose

def cargar_texturas(circuito, matrix_size):
    textura_tierra = cv2.imread('src/textura/tierra.jpg', cv2.IMREAD_COLOR)
    textura_asfalto = cv2.imread('src/textura/asfalto.jpg', cv2.IMREAD_COLOR)
    textura_tierra = cv2.resize(textura_tierra, (matrix_size, matrix_size))
    textura_asfalto = cv2.resize(textura_asfalto, (matrix_size, matrix_size))

    # Crear una máscara para la línea de meta
    linea_meta_mask = np.zeros(circuito.shape, dtype=np.uint8)
    cv2.line(linea_meta_mask, finish_line_start, finish_line_end, 255, thickness=2)

    # Aplicar la línea de meta a la textura del circuito
    circuito_texture = np.where(circuito[..., None] == 1, textura_asfalto, textura_tierra)
    circuito_texture[linea_meta_mask == 255] = [255, 255, 255]  # Pintar la línea de meta en blanco
    circuito_texture_rgba = cv2.cvtColor(circuito_texture, cv2.COLOR_BGR2RGBA)

    return circuito_texture_rgba

# Obtener la hora actual del sistema
current_hour = datetime.now().hour

# Definir el modelo de coche y la luz ambiental según la hora
if 8 <= current_hour < 14:
    coche_model_path = "src/3D/aston_martin_f1_amr23_2023.glb" # Aston martin si la hora es entre las 8 y las 14
    ambient_light = (1.0, 1.0, 1.0)  # Luz brillante
else:
    coche_model_path = "src/3D/oracle_red_bull_f1_car_rb19_2023.glb" # RedBull para las horas restantes
    ambient_light = (0.2, 0.2, 0.2)  # Luz más oscura

camIdMovil = 2
camId = 0

# Crear la escena con la luz ambiental ajustada
escena = pyrender.Scene(bg_color=(0, 0, 0), ambient_light=ambient_light)
matrix_size = 150

# Crear los circuitos
circuito1 = np.zeros((matrix_size, matrix_size), dtype=np.uint8)
cv2.circle(circuito1, (matrix_size // 2, matrix_size // 2), matrix_size // 3, 1, thickness=20)

finish_line_start_1 = (matrix_size // 2 + matrix_size // 3 - 10, matrix_size // 2)
finish_line_end_1 = (matrix_size // 2 + matrix_size // 3 + 10, matrix_size // 2)
cv2.line(circuito1, finish_line_start_1, finish_line_end_1, 1, thickness=2)

def crear_circuito_doble_o(matrix_size):
    circuito = np.zeros((matrix_size, matrix_size), dtype=np.uint8)
    cv2.ellipse(circuito, (matrix_size // 4, matrix_size // 2), (matrix_size // 6, matrix_size // 4), 0, 0, 360, 1, thickness=20)
    cv2.ellipse(circuito, (3 * matrix_size // 4, matrix_size // 2), (matrix_size // 6, matrix_size // 4), 0, 0, 360, 1, thickness=20)
    cv2.line(circuito, (matrix_size // 2 - matrix_size // 6, matrix_size // 2), (matrix_size // 2 + matrix_size // 6, matrix_size // 2), 1, thickness=20)
    return circuito

circuito2 = crear_circuito_doble_o(matrix_size)
finish_line_start_2 = (3 * matrix_size // 4 + matrix_size // 6 - 10, matrix_size // 2)
finish_line_end_2 = (3 * matrix_size // 4 + matrix_size // 6 + 10, matrix_size // 2)
cv2.line(circuito2, finish_line_start_2, finish_line_end_2, 1, thickness=2)


# Posiciones iniciales para cada nivel
initial_pos_world_1 = (0.30, -0.05, 0.0)
initial_pos_world_2 = (0.40, -0.05, 0.0)

# Inicialmente usar el circuito 1 y su posición inicial
circuito = circuito1
finish_line_start = finish_line_start_1
finish_line_end = finish_line_end_1
initial_pos_world = initial_pos_world_1

# Cargar las texturas del circuito
circuito_texture_rgba = cargar_texturas(circuito, matrix_size)

vertices = np.array([
    [-0.5, -0.5, 0.0],
    [0.5, -0.5, 0.0],
    [0.5, 0.5, 0.0],
    [-0.5, 0.5, 0.0]
])
faces = np.array([[0, 1, 2], [2, 3, 0]])
texture_coords = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
visual = trimesh.visual.texture.TextureVisuals(uv=texture_coords, image=circuito_texture_rgba)
mesh.visual = visual

modelo_circuito_mesh = pyrender.Mesh.from_trimesh(mesh)

escala_circuito = 1
mat_loc_circuito = mathutils.Matrix.Translation((0.0, 0.0, -0.02))
mat_rot_circuito = mathutils.Matrix.Rotation(math.radians(0.0), 4, 'X')
mat_sca_circuito = mathutils.Matrix.Scale(escala_circuito, 4)
meshpose_circuito = mat_loc_circuito @ mat_rot_circuito @ mat_sca_circuito

modelo_circuito = pyrender.Node(mesh=modelo_circuito_mesh, matrix=np.array(meshpose_circuito))
escena.add_node(modelo_circuito)

# Cargar el modelo del coche según la hora del sistema
with open(coche_model_path, 'rb') as f:
    modelo_coche_trimesh = trimesh.load(f, file_type='glb')
modelo_coche_mesh = pyrender.Mesh.from_trimesh(list(modelo_coche_trimesh.geometry.values()))

# Configurar las cámaras
bk = cuia.bestBackend(camIdMovil)
ar = cuia.myVideo(camIdMovil, bk)

bk_dir = cuia.bestBackend(camId)
ar_dir = cuia.myVideo(camId, bk_dir)

if not ar.isOpened() or not ar_dir.isOpened():
    print("Error: No se puede abrir alguna de las cámaras")
    exit()

# Crear instancia de CarControl pasando las cámaras y el tamaño de la matriz
car_control = CarControl(initial_pos_world, 0.0060, ar, ar_dir, matrix_size)

# Inicializar modelo del coche después de capturar el primer frame
first_frame_captured = False

fx = camara.cameraMatrix[0][0]
fy = camara.cameraMatrix[1][1]
cx = camara.cameraMatrix[0][2]
cy = camara.cameraMatrix[1][2]

camInt = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
cam = pyrender.Node(camera=camInt)
escena.add_node(cam)

hframe = int(ar.get(cv2.CAP_PROP_FRAME_HEIGHT))
wframe = int(ar.get(cv2.CAP_PROP_FRAME_WIDTH))

mirender = pyrender.OffscreenRenderer(wframe, hframe)

def realidadMixta(renderizador, frame, escena, vidas, lap_times, last_lap_time, nivel_superado):
    color, m = renderizador.render(escena)
    bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    _, m = cv2.threshold(m, 0, 1, cv2.THRESH_BINARY)
    m = (m * 255).astype(np.uint8)
    m = np.stack((m, m, m), axis=2)
    inversa = cv2.bitwise_not(m)
    pp = cv2.bitwise_and(bgr, m)
    fondo = cv2.bitwise_and(frame, inversa)
    res = cv2.bitwise_or(fondo, pp)
    
    # Dibujar el contador de vidas
    font = cv2.FONT_HERSHEY_COMPLEX
    if vidas > 0:
        cv2.putText(res, f'Vidas: {vidas}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(res, 'Vidas agotadas', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Dibujar el tiempo de vuelta
    if last_lap_time is not None:
        cv2.putText(res, f'Ultima vuelta: {last_lap_time:.2f}s', (wframe - 400, 60), font, 1, (0, 255, 255), 2, cv2.LINE_AA)  

    # Dibujar el tiempo transcurrido
    if car_control.start_time and not nivel_superado:
        elapsed_time = time.time() - car_control.start_time
        cv2.putText(res, f'Tiempo: {elapsed_time:.2f}s', (wframe - 400, 90), font, 1, (0, 255, 255), 2, cv2.LINE_AA) 
    
    # Dibujar el mensaje "Nivel Superado"
    if nivel_superado:
        cv2.putText(res, 'Nivel Superado', (wframe // 2 - 100, hframe // 2 - 20), font, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(res, f'Tiempo de vuelta: {last_lap_time:.2f}s', (wframe // 2 - 150, hframe // 2 + 20), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    
    return res

diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
detector = cv2.aruco.ArucoDetector(diccionario)

def mostrarModelo(frame):
    ret, rvec, tvec = car_control.detectarPose(frame, 5, 0.15, camIdMovil)
    if ret:
        poseCamara = fromOpencvToPyrender(rvec, tvec)
        escena.set_pose(cam, poseCamara)

    # Actualizar la posición del coche
    meshpose_coche = car_control.actualizarPosicion(circuito, matrix_size)
    escena.set_pose(modelo_coche, np.array(meshpose_coche))

    frame = realidadMixta(mirender, frame, escena, car_control.vidas, car_control.lap_times, car_control.last_lap_time, car_control.nivel_superado)
    return frame

def escuchar_comandos():
    while True:
        comando = car_control.escuchar_comando()
        if comando:
            if "nivel 1" in comando:
                cambiar_nivel(1)
            elif "nivel 2" in comando:
                cambiar_nivel(2)

def cambiar_nivel(nivel):
    global circuito, finish_line_start, finish_line_end, modelo_circuito_mesh, escena, modelo_circuito, car_control, initial_pos_world, modelo_coche

    escena.remove_node(modelo_circuito)

    if nivel == 1:
        circuito = circuito1
        finish_line_start = finish_line_start_1
        finish_line_end = finish_line_end_1
        initial_pos_world = initial_pos_world_1  # Posición inicial para el nivel 1
    elif nivel == 2:
        circuito = circuito2
        finish_line_start = finish_line_start_2
        finish_line_end = finish_line_end_2
        initial_pos_world = initial_pos_world_2  # Posición inicial para el nivel 2

    circuito_texture_rgba = cargar_texturas(circuito, matrix_size)

    visual = trimesh.visual.texture.TextureVisuals(uv=texture_coords, image=circuito_texture_rgba)
    mesh.visual = visual

    modelo_circuito_mesh = pyrender.Mesh.from_trimesh(mesh)
    modelo_circuito = pyrender.Node(mesh=modelo_circuito_mesh, matrix=np.array(meshpose_circuito))
    escena.add_node(modelo_circuito)

    # Reiniciar el estado del coche y del juego
    car_control.initial_pos_world = initial_pos_world  # Actualizar la posición inicial del coche
    car_control.pos_coche = np.array(initial_pos_world[:2])
    car_control.rot_coche = 0.0  # Reiniciar la rotación del coche
    car_control.vidas = car_control.vidas_iniciales  # Reiniciar vidas
    car_control.lap_times = []  # Reiniciar los tiempos de las vueltas
    car_control.start_time = None  # Reiniciar el tiempo de inicio
    car_control.last_lap_time = None  # Reiniciar el tiempo de la última vuelta
    car_control.nivel_superado = False  # Reiniciar el estado de nivel superado
    car_control.crossed_start_line = False  # Reiniciar la línea de meta cruzada
    car_control.cruzando_meta = False  # Reiniciar el cruce de meta

    # Actualizar la posición del coche en la escena
    meshpose_coche = car_control.actualizarPosicion(circuito, matrix_size)
    escena.remove_node(modelo_coche)
    modelo_coche = pyrender.Node(mesh=modelo_coche_mesh, matrix=np.array(meshpose_coche))
    escena.add_node(modelo_coche)

# Iniciar un hilo para escuchar comandos de voz
hilo_voz = threading.Thread(target=escuchar_comandos, daemon=False)
hilo_voz.start()

# Ciclo principal del programa
while ar.isOpened():
    ret, frame = ar.read()
    ret_dir, frame_dir = ar_dir.read()
    if not ret or not ret_dir:
        break

    # Capturar el frame de la cámara ID 0 para el control del coche si no se ha hecho aún
    if not first_frame_captured:
        meshpose_coche = car_control.actualizarPosicion(circuito, matrix_size)
        modelo_coche = pyrender.Node(mesh=modelo_coche_mesh, matrix=np.array(meshpose_coche))
        escena.add_node(modelo_coche)
        first_frame_captured = True

    frame = mostrarModelo(frame)
    cv2.imshow('AR', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ar.release()
ar_dir.release()
cv2.destroyAllWindows()

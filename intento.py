import os
import sys
import signal
import cv2
import torch
import numpy as np
import logging as log
import time
import serial
import math
import pyzed.sl as sl
from ultralytics import YOLO
from flask import Flask, Response

log.basicConfig(level=log.INFO)

# ==========================================
# 0. AJUSTES GENERALES, PUERTO SERIAL Y CONTROL
# ==========================================
TARGET_FPS = 15                 
NIVEL_DE_CONFIANZA = 0.40       
ID_CLASE_LATA = 0 
UMBRAL_PORCENTAJE_AMARILLO = 0.10

# --- Parámetros de Navegación y PID (Control P) ---
LIMITE_PWM = 110         # Límite máximo de velocidad (aprox 65% de 255)
MIN_PWM_GIRO = 80         # PWM mínimo para vencer inercia de motores al girar suave
KP_ROTACION = 250         # Constante Proporcional (K) para el giro
TOLERANCIA_X = 0.10       # Rango muerto para considerar centrada la lata (+/- 0.10m)

# Valores de color para detección
LOWER_YELLOW = np.array([16, 83, 109])
UPPER_YELLOW = np.array([30, 255, 255])

# Iniciar conexión serial
PUERTO_SERIAL = '/dev/ttyUSB0'
BAUDIOS = 115200
try:
    ser = serial.Serial(PUERTO_SERIAL, BAUDIOS, timeout=0.1)
    time.sleep(2) # Esperar reinicio de ESP32
    ser.reset_input_buffer()
    log.info("Puerto serial abierto correctamente.")
except serial.SerialException as e:
    log.error(f"Error al abrir el puerto serial: {e}")
    ser = None

# ==========================================
# 1. FUNCIONES DE CONTROL DE MOTORES
# ==========================================
def enviar_y_leer(m1, m2, m3, m4):
    """Aplica correcciones físicas y envía a la ESP32."""
    if ser is None or not ser.is_open:
        return

    # Inversión física de M1 y M4
    m1_final = m1 * -1
    m2_final = m2       
    m3_final = m3       
    m4_final = m4 * -1

    comando = f"{int(m1_final)},{int(m2_final)},{int(m3_final)},{int(m4_final)}\n"
    ser.write(comando.encode('utf-8'))
    
    if ser.in_waiting > 0:
        try:
            respuesta = ser.readline().decode('utf-8').strip()
            if respuesta:
                print(f"Encoders -> {respuesta} | Cmd: [{int(m1)},{int(m2)},{int(m3)},{int(m4)}]")
        except UnicodeDecodeError:
            pass

# ==========================================
# 2. MANEJO DE APAGADO SEGURO
# ==========================================
keep_running = True

def apagar_sistema(sig, frame):
    global keep_running
    print("\n[INFO] Apagando el sistema de visión...")
    keep_running = False
    if ser and ser.is_open:
        enviar_y_leer(0, 0, 0, 0) # Detener motores
        time.sleep(0.1)
        ser.close()
        print("Puerto serial cerrado.")
    try:
        zed.disable_positional_tracking()
        zed.close()
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, apagar_sistema)

# ==========================================
# 3. INFERENCIA YOLO (Modelo OBB)
# ==========================================
class ObjectDetection:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.object_model = YOLO("best.pt", task='obb')

    def inference(self, image: np.ndarray):
        return self.object_model.predict(image, conf=NIVEL_DE_CONFIANZA, classes=[ID_CLASE_LATA], device=self.device, verbose=False)

# ==========================================
# 4. CONFIGURACIÓN ZED E IMU
# ==========================================
app = Flask(__name__)
detector = ObjectDetection()

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA 
init_params.camera_fps = TARGET_FPS 
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.ULTRA 

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    log.error("No se pudo abrir la ZED.")
    sys.exit(1)

tracking_params = sl.PositionalTrackingParameters()
zed.enable_positional_tracking(tracking_params)

def dibujar_ejes_imu(frame, rot_matrix):
    cx, cy = 100, 150  
    largo = 40         
    x_axis = np.dot(rot_matrix, np.array([1, 0, 0])) * largo
    y_axis = np.dot(rot_matrix, np.array([0, -1, 0])) * largo
    z_axis = np.dot(rot_matrix, np.array([0, 0, -1])) * largo

    cv2.line(frame, (cx, cy), (int(cx + x_axis[0]), int(cy + x_axis[1])), (0, 0, 255), 3) 
    cv2.line(frame, (cx, cy), (int(cx + y_axis[0]), int(cy + y_axis[1])), (0, 255, 0), 3) 
    cv2.line(frame, (cx, cy), (int(cx + z_axis[0]), int(cy + z_axis[1])), (255, 0, 0), 3) 

def generar_video():
    image_zed = sl.Mat()
    point_cloud = sl.Mat() 
    sensors_data = sl.SensorsData()
    runtime_parameters = sl.RuntimeParameters()
    prev_time = 0
    global keep_running
    
    while keep_running:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ) 
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)

            pc_data = point_cloud.get_data()
            h_img, w_img = pc_data.shape[:2]
            rot_matrix = sensors_data.get_imu_data().get_pose().get_rotation_matrix().r
            temp_camara = sensors_data.get_temperature_data().get(sl.SENSOR_LOCATION.IMU)

            frame_bgr = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            results = detector.inference(frame_bgr)
            annotated_frame = results[0].plot(labels=False) 

            distancia_minima = float('inf')
            objetivo_x = None
            objetivo_z = None
            lata_detectada = False

            if results[0].obb is not None:
                xywhr_array = results[0].obb.xywhr.cpu().numpy()
                xyxyxyxy_array = results[0].obb.xyxyxyxy.cpu().numpy()
                
                if len(xywhr_array) > 0:
                    lata_detectada = True

                for i in range(len(xywhr_array)):
                    cx, cy = int(xywhr_array[i][0]), int(xywhr_array[i][1])
                    
                    y_inicio, y_fin = max(0, cy - 10), min(h_img, cy + 10)
                    x_inicio, x_fin = max(0, cx - 10), min(w_img, cx + 10)
                    
                    parche_x = pc_data[y_inicio:y_fin, x_inicio:x_fin, 0]
                    parche_z = pc_data[y_inicio:y_fin, x_inicio:x_fin, 2]
                    
                    valores_validos_x = parche_x[np.isfinite(parche_x)]
                    valores_validos_z = parche_z[np.isfinite(parche_z)]
                    
                    coord_x = None
                    coord_z = None

                    if len(valores_validos_x) > 0 and len(valores_validos_z) > 0:
                        coord_x = np.median(valores_validos_x)
                        coord_z = np.median(valores_validos_z)
                        str_coords = f"X:{coord_x:.2f}m Z:{coord_z:.2f}m"
                        
                        distancia_actual = math.hypot(coord_x, coord_z)
                        if distancia_actual < distancia_minima:
                            distancia_minima = distancia_actual
                            objetivo_x = coord_x
                            objetivo_z = coord_z
                    else:
                        str_coords = "Coords: Ciegas"

                    pts = xyxyxyxy_array[i]
                    x1, y1 = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
                    cv2.putText(annotated_frame, f"LATA | {str_coords}", (x1, max(20, y1 - 5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # --- LÓGICA DE NAVEGACIÓN Y CONTROL PROPORCIONAL ---
            v_m1, v_m2, v_m3, v_m4 = 0, 0, 0, 0 

            if lata_detectada and objetivo_x is not None and objetivo_z is not None:
                error_x = abs(objetivo_x)
                
                if objetivo_x > TOLERANCIA_X:
                    # Lata a la derecha -> Girar derecha progresivamente
                    # Calcula velocidad: base mínima + (error * constante)
                    vel_giro = int(MIN_PWM_GIRO + (error_x * KP_ROTACION))
                    vel_giro = min(LIMITE_PWM, vel_giro) # Nunca exceder el 65%
                    
                    v_m1, v_m2 = vel_giro, vel_giro
                    v_m3, v_m4 = -vel_giro, -vel_giro
                    
                elif objetivo_x < -TOLERANCIA_X:
                    # Lata a la izquierda -> Girar izquierda progresivamente
                    vel_giro = int(MIN_PWM_GIRO + (error_x * KP_ROTACION))
                    vel_giro = min(LIMITE_PWM, vel_giro)
                    
                    v_m1, v_m2 = -vel_giro, -vel_giro
                    v_m3, v_m4 = vel_giro, vel_giro
                    
                else:
                    # Lata centrada (dentro de +/- 0.10) -> Avanzar recto
                    v_m1, v_m2 = LIMITE_PWM, LIMITE_PWM
                    v_m3, v_m4 = LIMITE_PWM, LIMITE_PWM
            
            # Enviar comandos a la ESP32
            enviar_y_leer(v_m1, v_m2, v_m3, v_m4)

            # --- HUD E IMU VISUAL ---
            cv2.rectangle(annotated_frame, (5, 5), (300, 50), (0, 0, 0), -1)
            cv2.putText(annotated_frame, f"FPS: {int(fps)} | Temp: {temp_camara:.1f}C", (15, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            dibujar_ejes_imu(annotated_frame, rot_matrix)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generar_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='172.23.254.220', port=5000)
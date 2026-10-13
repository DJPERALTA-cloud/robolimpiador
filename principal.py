import os
import sys
import signal
import cv2
import torch
import numpy as np
import logging as log
import time
import pyzed.sl as sl
from ultralytics import YOLO
from flask import Flask, Response

log.basicConfig(level=log.INFO)

# ==========================================
# 0. AJUSTES GENERALES
# ==========================================
TARGET_FPS = 15                 
NIVEL_DE_CONFIANZA = 0.25      
ID_CLASE_LATA = 0 

# ==========================================
# 1. MANEJO DE APAGADO SEGURO
# ==========================================
keep_running = True

def apagar_sistema(sig, frame):
    global keep_running
    print("\n[INFO] Apagando el sistema de visión...")
    keep_running = False
    try:
        zed.disable_positional_tracking()
        zed.close()
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, apagar_sistema)

# ==========================================
# 2. INFERENCIA YOLO (Modelo OBB)
# ==========================================
class ObjectDetection:
    def __init__(self):
        # Aseguramos el uso de GPU con device=0
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.object_model = YOLO("best.pt", task='obb')

    def inference(self, image: np.ndarray):
        return self.object_model.predict(image, conf=NIVEL_DE_CONFIANZA, classes=[ID_CLASE_LATA], device=self.device, verbose=False)

# ==========================================
# 3. CONFIGURACIÓN ZED E IMU
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
    depth_map = sl.Mat()
    sensors_data = sl.SensorsData()
    # La profundidad siempre estará activa por defecto en el Runtime
    runtime_parameters = sl.RuntimeParameters()
    prev_time = 0
    global keep_running
    
    while keep_running:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)

            depth_array = depth_map.get_data()
            h_img, w_img = depth_array.shape
            rot_matrix = sensors_data.get_imu_data().get_pose().get_rotation_matrix().r
            temp_camara = sensors_data.get_temperature_data().get(sl.SENSOR_LOCATION.IMU)

            frame_bgr = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # INFERENCIA YOLO
            results = detector.inference(frame_bgr)
            annotated_frame = results[0].plot()

            # CORRECCIÓN PARA MODELO OBB
            if results[0].obb is not None:
                # Extraemos el formato xywhr (centro_x, centro_y, width, height, rotación)
                for obb in results[0].obb.xywhr.cpu().numpy():
                    cx, cy = int(obb[0]), int(obb[1])
                    
                    # Filtro anti-reflejos tomando un área de 20x20 píxeles alrededor del centro
                    y_inicio, y_fin = max(0, cy - 10), min(h_img, cy + 10)
                    x_inicio, x_fin = max(0, cx - 10), min(w_img, cx + 10)
                    parche = depth_array[y_inicio:y_fin, x_inicio:x_fin]
                    valores_validos = parche[np.isfinite(parche)]
                    
                    if len(valores_validos) > 0:
                        distancia_z = np.median(valores_validos)
                        texto = f"Distancia: {distancia_z:.2f}m"
                        color_texto = (0, 255, 0)
                    else:
                        texto = "Distancia: Ciego"
                        color_texto = (0, 0, 255)
                        
                    cv2.putText(annotated_frame, texto, (cx, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)

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
    app.run(host='192.168.55.1', port=5000)
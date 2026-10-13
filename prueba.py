import serial
import time
import threading
import torch
import numpy as np
import math
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from flask import Flask, Response

# ==============================================================
# SISTEMA DE VISIÓN (integrado desde vision_system.py)
# ==============================================================
class VisionSystem:
    def __init__(self):
        self.TARGET_FPS = 15
        self.NIVEL_DE_CONFIANZA = 0.40
        self.ID_CLASE_LATA = 0
        self.LOWER_YELLOW = np.array([16, 83, 109])
        self.UPPER_YELLOW = np.array([30, 255, 255])

        self.lata_detectada = False
        self.objetivo_x = None
        self.objetivo_z = None
        self.color_lata = "Amarillo"
        self.latest_frame = None
        self.keep_running = True

        # Ancho del frame de la cámara (se actualiza en el loop)
        self.frame_width = 1

        # Guardar el ancho máximo del bounding box más cercano (en píxeles)
        self.bbox_width_px = 0

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.object_model = YOLO("best.pt", task='obb')

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA
        init_params.camera_fps = self.TARGET_FPS
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("[ERROR] No se pudo abrir la cámara ZED.")
            exit(1)

        tracking_params = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(tracking_params)

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        self.app = Flask(__name__)

        @self.app.route('/video')
        def video_feed():
            return Response(self._generar_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _dibujar_ejes_imu(self, frame, rot_matrix):
        cx, cy = 100, 150
        largo = 40
        x_axis = np.dot(rot_matrix, np.array([1, 0, 0])) * largo
        y_axis = np.dot(rot_matrix, np.array([0, -1, 0])) * largo
        z_axis = np.dot(rot_matrix, np.array([0, 0, -1])) * largo
        cv2.line(frame, (cx, cy), (int(cx + x_axis[0]), int(cy + x_axis[1])), (0, 0, 255), 3)
        cv2.line(frame, (cx, cy), (int(cx + y_axis[0]), int(cy + y_axis[1])), (0, 255, 0), 3)
        cv2.line(frame, (cx, cy), (int(cx + z_axis[0]), int(cy + z_axis[1])), (255, 0, 0), 3)

    def _capture_loop(self):
        image_zed = sl.Mat()
        point_cloud = sl.Mat()
        sensors_data = sl.SensorsData()
        runtime_parameters = sl.RuntimeParameters()
        prev_time = 0

        while self.keep_running:
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
                self.zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)

                pc_data = point_cloud.get_data()
                h_img, w_img = pc_data.shape[:2]
                rot_matrix = sensors_data.get_imu_data().get_pose().get_rotation_matrix().r
                temp_camara = sensors_data.get_temperature_data().get(sl.SENSOR_LOCATION.IMU)

                frame_bgr = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
                self.frame_width = frame_bgr.shape[1]  # Actualizar ancho real del frame

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time

                results = self.object_model.predict(
                    frame_bgr, conf=self.NIVEL_DE_CONFIANZA,
                    classes=[self.ID_CLASE_LATA], device=self.device, verbose=False
                )
                annotated_frame = results[0].plot(labels=False)

                distancia_minima = float('inf')
                temp_objetivo_x = None
                temp_objetivo_z = None
                temp_lata_detectada = False
                temp_bbox_width_px = 0  # Ancho del bbox de la lata más cercana

                if results[0].obb is not None:
                    xywhr_array = results[0].obb.xywhr.cpu().numpy()
                    xyxyxyxy_array = results[0].obb.xyxyxyxy.cpu().numpy()

                    if len(xywhr_array) > 0:
                        temp_lata_detectada = True

                    for i in range(len(xywhr_array)):
                        cx_det, cy_det = int(xywhr_array[i][0]), int(xywhr_array[i][1])

                        y_inicio = max(0, cy_det - 10)
                        y_fin    = min(h_img, cy_det + 10)
                        x_inicio = max(0, cx_det - 10)
                        x_fin    = min(w_img, cx_det + 10)

                        parche_x = pc_data[y_inicio:y_fin, x_inicio:x_fin, 0]
                        parche_z = pc_data[y_inicio:y_fin, x_inicio:x_fin, 2]

                        valores_validos_x = parche_x[np.isfinite(parche_x)]
                        valores_validos_z = parche_z[np.isfinite(parche_z)]

                        coord_x, coord_z = None, None

                        if len(valores_validos_x) > 0 and len(valores_validos_z) > 0:
                            coord_x = np.median(valores_validos_x)
                            coord_z = np.median(valores_validos_z)
                            str_coords = f"X:{coord_x:.2f}m Z:{coord_z:.2f}m"

                            distancia_actual = math.hypot(coord_x, coord_z)
                            if distancia_actual < distancia_minima:
                                distancia_minima = distancia_actual
                                temp_objetivo_x = coord_x
                                temp_objetivo_z = coord_z

                                # Calcular ancho del bbox en píxeles para la lata más cercana
                                pts = xyxyxyxy_array[i]
                                x_max_bbox = int(np.max(pts[:, 0]))
                                x_min_bbox = int(np.min(pts[:, 0]))
                                temp_bbox_width_px = x_max_bbox - x_min_bbox
                        else:
                            str_coords = "Coords: Ciegas"

                        pts = xyxyxyxy_array[i]
                        x1 = int(np.min(pts[:, 0]))
                        y1 = int(np.min(pts[:, 1]))
                        cv2.putText(
                            annotated_frame, f"LATA | {str_coords}",
                            (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                        )

                # Actualizar estado compartido
                self.lata_detectada   = temp_lata_detectada
                self.objetivo_x       = temp_objetivo_x
                self.objetivo_z       = temp_objetivo_z
                self.bbox_width_px    = temp_bbox_width_px

                # Indicador visual de proximidad crítica en el frame
                if self.frame_width > 0:
                    ratio = self.bbox_width_px / self.frame_width
                    if ratio >= 0.75:
                        cv2.putText(
                            annotated_frame, "!! STOP: LATA MUY CERCA !!",
                            (10, annotated_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                        )

                # HUD
                cv2.rectangle(annotated_frame, (5, 5), (300, 50), (0, 0, 0), -1)
                cv2.putText(
                    annotated_frame, f"FPS: {int(fps)} | Temp: {temp_camara:.1f}C",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                self._dibujar_ejes_imu(annotated_frame, rot_matrix)
                self.latest_frame = annotated_frame

    def _generar_video(self):
        while self.keep_running:
            if self.latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.latest_frame)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)

    def start_server(self):
        threading.Thread(
            target=lambda: self.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False),
            daemon=True
        ).start()

    def lata_muy_cerca(self, umbral=0.75):
        """
        Devuelve True si el bbox de la lata más cercana supera
        el porcentaje 'umbral' del ancho total del frame.
        """
        if self.frame_width <= 0:
            return False
        ratio = self.bbox_width_px / self.frame_width
        return ratio >= umbral

    def get_data(self):
        return self.lata_detectada, self.objetivo_x, self.objetivo_z, self.color_lata

    def stop(self):
        self.keep_running = False
        try:
            self.zed.disable_positional_tracking()
            self.zed.close()
        except:
            pass


# ==============================================================
# SISTEMA DE MOVIMIENTO
# ==============================================================
PUERTO  = '/dev/ttyUSB0'
BAUDIOS = 115200
PWM_MAX = 100
UMBRAL_PROXIMIDAD = 0.75  # 75% del ancho del frame

try:
    ser = serial.Serial(PUERTO, BAUDIOS, timeout=0.1)
    time.sleep(2)
    ser.reset_input_buffer()
    print(f"Conectado a {PUERTO}. Velocidad configurada a {PWM_MAX} PWM.")
except serial.SerialException as e:
    print(f"Error al abrir el puerto: {e}")
    exit()


def enviar_movimiento(m1, m2, m3, m4):
    m1_final = m1 * -1
    m2_final = m2
    m3_final = m3
    m4_final = m4 * -1
    comando = f"{m1_final},{m2_final},{m3_final},{m4_final}\n"
    ser.write(comando.encode('utf-8'))


def mover_robot(direccion, milisegundos, vision: VisionSystem = None):
    """
    Mueve el robot en la dirección indicada durante 'milisegundos'.
    Si 'vision' está disponible y la dirección es 'w' (adelante),
    se detiene automáticamente cuando la lata ocupa más del 75% del frame.
    """
    if direccion == 'w':
        izq, der = PWM_MAX, PWM_MAX
    elif direccion == 's':
        izq, der = -PWM_MAX, -PWM_MAX
    elif direccion == 'a':
        izq, der = -PWM_MAX, PWM_MAX
    elif direccion == 'd':
        izq, der = PWM_MAX, -PWM_MAX
    else:
        print("Dirección no válida.")
        return

    duracion_segundos = milisegundos / 1000.0
    start_time = time.time()
    detenido_por_vision = False

    while time.time() - start_time < duracion_segundos:

        # Guardia de proximidad: solo activa al avanzar
        if direccion == 'w' and vision is not None and vision.lata_muy_cerca(UMBRAL_PROXIMIDAD):
            print(f"[VISIÓN] Lata detectada a menos del umbral ({int(UMBRAL_PROXIMIDAD*100)}% del frame). ¡STOP!")
            detenido_por_vision = True
            break

        enviar_movimiento(izq, izq, der, der)
        time.sleep(0.05)  # ~20 Hz

    enviar_movimiento(0, 0, 0, 0)

    if detenido_por_vision:
        tiempo_restante = milisegundos - int((time.time() - start_time) * 1000)
        print(f"[INFO] Movimiento interrumpido. Quedaban ~{tiempo_restante} ms.")


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    import serial  # Asegurar import dentro del bloque principal

    print("[INIT] Iniciando sistema de visión...")
    vision = VisionSystem()
    vision.start_server()
    print("[INIT] Servidor de video disponible en http://<IP>:5000/video")
    print("[INIT] Sistema listo.\n")

    try:
        print("--- Control del Robot Activo ---")
        print("Controles: w (adelante), s (atrás), a (izquierda), d (derecha)")
        print("Formato: [tecla] [milisegundos]  →  Ejemplo: w 4000")
        print("Ctrl+C para salir.\n")

        while True:
            entrada = input("Comando >> ").strip().lower().split()

            if len(entrada) == 2:
                tecla = entrada[0]
                try:
                    tiempo_ms = int(entrada[1])
                    mover_robot(tecla, tiempo_ms, vision=vision)
                except ValueError:
                    print("Error: los milisegundos deben ser un número entero.")
            else:
                print("Error: usa el formato  [tecla] [milisegundos]")

    except KeyboardInterrupt:
        print("\nDeteniendo programa...")

    finally:
        for _ in range(5):
            try:
                ser.write(b"0,0,0,0\n")
                time.sleep(0.01)
            except:
                pass
        ser.close()
        vision.stop()
        print("Puerto cerrado. Robot detenido.")
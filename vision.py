import cv2
import torch
import numpy as np
import math
import time
import threading
import pyzed.sl as sl
from ultralytics import YOLO
from flask import Flask, Response

DIST_INFINITA = 100.0

class VisionSystem:
    def __init__(self):
        self.TARGET_FPS = 20
        self.NIVEL_DE_CONFIANZA = 0.15
        self.ID_CLASE_LATA = 0

        # Parámetros detección de bordes (azules)
        self.LOWER_BORDE = np.array([104, 73, 50])
        self.UPPER_BORDE = np.array([127, 224, 194])
        self.AREA_MINIMA_BORDE = 1000
        
        # Parámetros detección del objetivo (aro rojo)
        self.LOWER_ROJO = np.array([142, 105, 45])
        self.UPPER_ROJO = np.array([179, 217, 129])
        self.AREA_MINIMA_OBJETIVO = 100
        
        self.FRAMES_ENTRE_BORDES = 3 
        self._frame_contador = 0

        # Variables compartidas - latas
        self.lata_detectada = False
        self.objetivo_x = None
        self.objetivo_z = None
        self.color_lata = "Amarillo"

        # Variables compartidas - bordes
        self.dist_borde_izq = DIST_INFINITA
        self.dist_borde_cen = DIST_INFINITA
        self.dist_borde_der = DIST_INFINITA
        self.last_contornos = [] 

        # Variables compartidas - Objetivo Rojo
        self.target_detectado = False
        self.target_x = None
        self.target_z = None
        self.last_target_contour = None

        self.latest_frame = None
        self.keep_running = True

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

    def _detectar_objetivo_rojo(self, frame_bgr, pc_data):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_ROJO, self.UPPER_ROJO)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mejor_contorno = None
        mayor_area = 0
        obj_x, obj_z = None, None
        detectado = False

        if contours:
            for c in contours:
                area = cv2.contourArea(c)
                if area > self.AREA_MINIMA_OBJETIVO:
                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    
                    if hull_area > 0:
                        solidity = float(area) / hull_area
                        # Un aro o marco hueco tiene una solidez baja. 
                        # Un bloque sólido rojo tendrá solidez > 0.9
                        if solidity < 2.5: 
                            if area > mayor_area:
                                mayor_area = area
                                mejor_contorno = c

        if mejor_contorno is not None:
            # Crear una máscara vacía para aislar solo este contorno
            c_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(c_mask, [mejor_contorno], -1, 255, -1)
            
            # Extraer profundidad SOLO de los píxeles que son rojos Y están dentro del contorno
            # Esto evita medir la arena que queda en el centro del aro
            z_data = pc_data[:, :, 2]
            x_data = pc_data[:, :, 0]
            valid_pixels = (c_mask == 255) & (mask == 255) & (z_data > 0) & np.isfinite(z_data)

            if np.any(valid_pixels):
                obj_z = float(np.median(z_data[valid_pixels]))
                obj_x = float(np.median(x_data[valid_pixels]))
                detectado = True

        return detectado, obj_x, obj_z, mejor_contorno

    def _detectar_bordes(self, frame_bgr, pc_data):
        h_img, w_img = frame_bgr.shape[:2]
        w_tercio = w_img // 3

        d_izq = DIST_INFINITA
        d_cen = DIST_INFINITA
        d_der = DIST_INFINITA

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_BORDE, self.UPPER_BORDE)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_validos = [c for c in contours if cv2.contourArea(c) > self.AREA_MINIMA_BORDE]

        if not contornos_validos:
            return d_izq, d_cen, d_der, []

        puntos = np.vstack(contornos_validos).squeeze(1)
        xs = np.clip(puntos[:, 0], 0, w_img - 1)
        ys = np.clip(puntos[:, 1], 0, h_img - 1)

        zs = pc_data[ys, xs, 2]

        validos = (zs > 0) & np.isfinite(zs)
        xs_val = xs[validos]
        zs_val = zs[validos]

        if len(zs_val) > 0:
            mask_izq = xs_val < w_tercio
            mask_cen = (xs_val >= w_tercio) & (xs_val < 2 * w_tercio)
            mask_der = xs_val >= 2 * w_tercio

            if np.any(mask_izq): d_izq = float(np.min(zs_val[mask_izq]))
            if np.any(mask_cen): d_cen = float(np.min(zs_val[mask_cen]))
            if np.any(mask_der): d_der = float(np.min(zs_val[mask_der]))

        return d_izq, d_cen, d_der, contornos_validos

    def _capture_loop(self):
        image_zed = sl.Mat()
        point_cloud = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        prev_time = 0

        while self.keep_running:
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

                pc_data = point_cloud.get_data()
                h_img, w_img = pc_data.shape[:2]

                frame_bgr = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)

                altura_mancha = 150
                cv2.rectangle(frame_bgr, (0, 0), (w_img, altura_mancha), (0, 0, 0), -1)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time

                # 1. Detección de Latas (YOLO)
                results = self.object_model.predict(frame_bgr, conf=self.NIVEL_DE_CONFIANZA, classes=[self.ID_CLASE_LATA], device=self.device, verbose=False)
                annotated_frame = results[0].plot(labels=False)

                distancia_minima = float('inf')
                temp_objetivo_x = None
                temp_objetivo_z = None
                temp_lata_detectada = False

                if results[0].obb is not None:
                    xywhr_array = results[0].obb.xywhr.cpu().numpy()
                    xyxyxyxy_array = results[0].obb.xyxyxyxy.cpu().numpy()

                    if len(xywhr_array) > 0: temp_lata_detectada = True

                    for i in range(len(xywhr_array)):
                        cx, cy = int(xywhr_array[i][0]), int(xywhr_array[i][1])

                        y_inicio, y_fin = max(0, cy - 10), min(h_img, cy + 10)
                        x_inicio, x_fin = max(0, cx - 10), min(w_img, cx + 10)

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
                        else:
                            str_coords = "Coords: Ciegas"

                        pts = xyxyxyxy_array[i]
                        x1, y1 = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
                        cv2.putText(annotated_frame, f"LATA | {str_coords}", (x1, max(20, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                self.lata_detectada = temp_lata_detectada
                self.objetivo_x = temp_objetivo_x
                self.objetivo_z = temp_objetivo_z

                # 2. Detección del Objetivo Rojo (Cada frame)
                t_det, t_x, t_z, contorno_rojo = self._detectar_objetivo_rojo(frame_bgr, pc_data)
                self.target_detectado = t_det
                self.target_x = t_x
                self.target_z = t_z
                
                if contorno_rojo is not None:
                    # Dibujamos el contorno del objetivo en morado para distinguirlo
                    cv2.drawContours(annotated_frame, [contorno_rojo], -1, (255, 0, 255), 3)
                    if t_det:
                        M = cv2.moments(contorno_rojo)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.putText(annotated_frame, f"OBJETIVO: X:{t_x:.2f}m Z:{t_z:.2f}m", (cx - 50, cy - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # 3. Cálculo de bordes azules (1 de cada N frames)
                self._frame_contador += 1
                if self._frame_contador >= self.FRAMES_ENTRE_BORDES:
                    self._frame_contador = 0
                    d_izq, d_cen, d_der, contornos = self._detectar_bordes(frame_bgr, pc_data)
                    self.dist_borde_izq = d_izq
                    self.dist_borde_cen = d_cen
                    self.dist_borde_der = d_der
                    self.last_contornos = contornos

                if self.last_contornos:
                    cv2.drawContours(annotated_frame, self.last_contornos, -1, (0, 0, 255), 2)
                
                w_tercio = w_img // 3
                cv2.line(annotated_frame, (w_tercio, altura_mancha), (w_tercio, h_img), (255, 255, 0), 1)
                cv2.line(annotated_frame, (2 * w_tercio, altura_mancha), (2 * w_tercio, h_img), (255, 255, 0), 1)
                
                str_izq = f"IZQ:{self.dist_borde_izq:.1f}m" if self.dist_borde_izq < DIST_INFINITA else "IZQ:---"
                str_cen = f"CEN:{self.dist_borde_cen:.1f}m" if self.dist_borde_cen < DIST_INFINITA else "CEN:---"
                str_der = f"DER:{self.dist_borde_der:.1f}m" if self.dist_borde_der < DIST_INFINITA else "DER:---"
                
                cv2.putText(annotated_frame, str_izq, (10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_frame, str_cen, (w_tercio + 10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_frame, str_der, (2 * w_tercio + 10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.rectangle(annotated_frame, (5, 5), (120, 50), (0, 0, 0), -1)
                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                self.latest_frame = annotated_frame

    def _generar_video(self):
        while self.keep_running:
            if self.latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.latest_frame)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)

    def start_server(self):
        threading.Thread(target=lambda: self.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()

    def get_data(self):
        # Mantenemos las 7 variables originales para no romper el programa actual
        return (
            self.lata_detectada,
            self.objetivo_x,
            self.objetivo_z,
            self.color_lata,
            self.dist_borde_izq,
            self.dist_borde_cen,
            self.dist_borde_der
        )

    def get_target_data(self):
        # Nueva función exclusiva para leer los datos del aro rojo
        return (
            self.target_detectado,
            self.target_x,
            self.target_z
        )

    def stop(self):
        self.keep_running = False
        try:
            self.zed.disable_positional_tracking()
            self.zed.close()
        except:
            pass
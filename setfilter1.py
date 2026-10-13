import sys
import signal
import cv2
import torch
import numpy as np
import time
import logging as log
from ultralytics import YOLO
from flask import Flask, Response

log.basicConfig(level=log.INFO)

# ==========================================
# 0. AJUSTES GENERALES
# ==========================================
NIVEL_DE_CONFIANZA = 0.55       
ID_CLASE_LATA = 0 

# Valores de color obtenidos con los sliders
LOWER_YELLOW = np.array([16, 83, 109])
UPPER_YELLOW = np.array([30, 255, 255])

RUTA_VIDEO = 'video_prueba.mp4' 

# ==========================================
# 1. MANEJO DE APAGADO SEGURO
# ==========================================
keep_running = True

def apagar_sistema(sig, frame):
    global keep_running
    print("\n[INFO] Apagando el servidor de visión...")
    keep_running = False
    sys.exit(0)

signal.signal(signal.SIGINT, apagar_sistema)

# ==========================================
# 2. INFERENCIA YOLO
# ==========================================
class ObjectDetection:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.object_model = YOLO("best.pt").to(self.device)

    def inference(self, image: np.ndarray):
        return self.object_model(image, conf=NIVEL_DE_CONFIANZA, classes=[ID_CLASE_LATA], verbose=False)

# ==========================================
# 3. SERVIDOR FLASK Y PROCESAMIENTO
# ==========================================
app = Flask(__name__)
detector = ObjectDetection()

def generar_video():
    global keep_running
    cap = cv2.VideoCapture(RUTA_VIDEO)
    
    if not cap.isOpened():
        log.error(f"No se pudo abrir el video: {RUTA_VIDEO}")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    delay_frame = 1.0 / fps_video if fps_video > 0 else 0.033
    prev_time = 0

    while keep_running:
        start_process_time = time.time()
        ret, frame_bgr = cap.read()
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h_img, w_img, _ = frame_bgr.shape

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # INFERENCIA YOLO
        results = detector.inference(frame_bgr)
        annotated_frame = results[0].plot(labels=False)

        # ----------------------------------------------------
        # CORRECCIÓN: Manejo de OBB vs Bounding Boxes normales
        # ----------------------------------------------------
        if results[0].obb is not None:
            detecciones = results[0].obb
            es_obb = True
        else:
            detecciones = results[0].boxes
            es_obb = False

        if detecciones is not None and len(detecciones) > 0:
            for i in range(len(detecciones)):
                
                # 1. Obtener coordenadas dependiendo del tipo de modelo
                if es_obb:
                    # Si es OBB, sacamos las 4 esquinas y calculamos los extremos (min y max)
                    pts = detecciones.xyxyxyxy[i].cpu().numpy()
                    x1, x2 = np.min(pts[:, 0]), np.max(pts[:, 0])
                    y1, y2 = np.min(pts[:, 1]), np.max(pts[:, 1])
                else:
                    x1, y1, x2, y2 = detecciones.xyxy[i].cpu().numpy()
                    
                x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
                x2_int, y2_int = min(w_img, int(x2)), min(h_img, int(y2))
                
                # 2. Extraer la Región de Interés (ROI) recta alrededor de la lata
                roi = frame_bgr[y1_int:y2_int, x1_int:x2_int]
                
                if roi.size > 0:
                    # 3. Análisis de HSV
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask_yellow = cv2.inRange(hsv_roi, LOWER_YELLOW, UPPER_YELLOW)
                    
                    pixeles_amarillos = cv2.countNonZero(mask_yellow)
                    pixeles_totales = roi.shape[0] * roi.shape[1]
                    porcentaje_amarillo = pixeles_amarillos / pixeles_totales if pixeles_totales > 0 else 0
                    
                    # 4. Lógica de decisión (>10% amarillo)
                    if porcentaje_amarillo >= 0.10:
                        texto_final = f"LATA AMARILLA ({porcentaje_amarillo*100:.0f}%)"
                        color_letra = (0, 255, 255) 
                    else:
                        texto_final = f"LATA NEGRA ({porcentaje_amarillo*100:.0f}%)"
                        color_letra = (200, 200, 200) 
                        
                    # 5. Dibujar texto en la parte superior del cuadro
                    (w_texto, h_texto), _ = cv2.getTextSize(texto_final, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated_frame, (x1_int, max(0, y1_int - 25)), (x1_int + w_texto, max(0, y1_int)), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, texto_final, (x1_int, max(20, y1_int - 5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_letra, 2)

        # HUD con los FPS
        cv2.rectangle(annotated_frame, (5, 5), (150, 40), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret_enc, buffer = cv2.imencode('.jpg', annotated_frame)
        if ret_enc:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        process_time = time.time() - start_process_time
        sleep_time = delay_frame - process_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()

@app.route('/')
def video_feed():
    return Response(generar_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("\n[INFO] Servidor simulando cámara iniciado.")
    print("[INFO] Abre tu navegador web en: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)
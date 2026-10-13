import sys
import signal
import cv2
import math
import pyzed.sl as sl
from flask import Flask, Response

# ==========================================
# 1. MANEJO DE APAGADO SEGURO (Anti-Reboots)
# ==========================================
keep_running = True

def apagar_sistema(sig, frame):
    global keep_running
    print("\n\n[ALERTA] Apagando servidor...")
    keep_running = False
    try:
        # Apagamos los módulos de IA antes de cerrar la cámara
        zed.disable_object_detection()
        zed.disable_positional_tracking()
        zed.close()
        print("[ÉXITO] Cámara ZED 2i liberada correctamente.")
    except Exception as e:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, apagar_sistema)

# ==========================================
# 2. INICIALIZACIÓN DE LA CÁMARA ZED
# ==========================================
print("[INFO] Iniciando cámara ZED...")
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA
init_params.camera_fps = 15
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Error: No se pudo abrir la cámara ZED. Revisa el cable USB.")
    sys.exit(1)

# ==========================================
# 3. CONFIGURACIÓN DE IA NATIVA (ZED SDK 5.2)
# ==========================================
print("[INFO] Encendiendo Tracking Espacial (Requerido para IA)...")
tracking_params = sl.PositionalTrackingParameters()
zed.enable_positional_tracking(tracking_params)

print("[INFO] Cargando motor de Detección de Personas...")
obj_param = sl.ObjectDetectionParameters()
obj_param.enable_tracking = True # Permite seguir a la persona frame a frame (ID único)
obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST

if zed.enable_object_detection(obj_param) != sl.ERROR_CODE.SUCCESS:
    print("Error: No se pudo cargar el módulo de IA de la ZED.")
    zed.close()
    sys.exit(1)

# Filtramos para que SOLO detecte personas con más del 50% de confianza
obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
obj_runtime_param.detection_confidence_threshold = 50
obj_runtime_param.object_class_filter = [sl.OBJECT_CLASS.PERSON]

# ==========================================
# 4. SERVIDOR FLASK Y STREAMING
# ==========================================
app = Flask(__name__)

def generar_video():
    image_zed = sl.Mat()
    objects = sl.Objects()
    runtime_parameters = sl.RuntimeParameters()
    global keep_running
    
    while keep_running:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Recuperamos imagen y la lista de objetos detectados (Personas)
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_objects(objects, obj_runtime_param)

            # Convertimos a formato OpenCV
            frame_bgr = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)

            # --- DIBUJAR DETECCIONES NATIVAS ---
            if objects.is_new:
                obj_array = objects.object_list
                for obj in obj_array:
                    # Obtenemos las coordenadas de la caja (esquinas superior-izq e inferior-der)
                    pt1 = (int(obj.bounding_box_2d[0][0]), int(obj.bounding_box_2d[0][1]))
                    pt2 = (int(obj.bounding_box_2d[2][0]), int(obj.bounding_box_2d[2][1]))
                    
                    # MAGIA NATIVA: ZED ya calculó la distancia al centro de la persona por nosotros
                    distancia_z = obj.position[2]
                    
                    # Filtramos lecturas inválidas
                    if not math.isnan(distancia_z) and not math.isinf(distancia_z):
                        # Dibujamos caja delimitadora
                        cv2.rectangle(frame_bgr, pt1, pt2, (255, 100, 0), 2)
                        
                        # Preparamos el texto (Ej. "Persona 0: 1.52m")
                        texto = f"Persona {obj.id}: {distancia_z:.2f} m"
                        
                        # Fondo negro para el texto para que sea legible
                        (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame_bgr, (pt1[0], pt1[1] - 25), (pt1[0] + w, pt1[1]), (0, 0, 0), -1)
                        cv2.putText(frame_bgr, texto, (pt1[0], pt1[1] - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generar_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# 5. INICIO DEL PROGRAMA
# ==========================================
if __name__ == "__main__":
    print("\n=====================================================")
    print("¡Servidor ZED Nativo (Detección de Personas) en línea!")
    print("Abre tu navegador en la laptop y entra a tu IP en el puerto 5000")
    print("--> PARA SALIR PRESIONA: CTRL + C <--")
    print("=====================================================\n")
    app.run(host='0.0.0.0', port=5000)
import cv2
import numpy as np
import pyzed.sl as sl
from flask import Flask, Response
import threading
import serial
import time
import sys
import logging

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
LOWER_BLUE = np.array([104, 73, 50])
UPPER_BLUE = np.array([127, 224, 194])
latest_frame = None

app = Flask(__name__)

puerto = '/dev/ttyUSB0' 
baudios = 115200
PWM_MAX = 254
estado_pca = [-1] * 16
ser = None

# ==========================================
# RUTINAS DE VISIÓN Y STREAMING
# ==========================================
def capture_loop():
    global latest_frame
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 15
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL # Requerido para medir profundidad

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("[ERROR] No se pudo abrir la cámara ZED.")
        sys.exit(1)

    image_zed = sl.Mat()
    point_cloud = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ) # Extraer datos 3D
            
            frame_bgr = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
            pc_data = point_cloud.get_data()
            
            h_img, w_img = frame_bgr.shape[:2]
            
            # Máscara negra en el 40% superior
            altura_mancha = int(h_img * 0.4) 
            cv2.rectangle(frame_bgr, (0, 0), (w_img, altura_mancha), (0, 0, 0), -1)
            
            hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, LOWER_BLUE, UPPER_BLUE)
            
            # Limpieza morfológica (Ruido y uniones)
            kernel_open = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            kernel_close = np.ones((20, 20), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contornos_validos = [c for c in contours if cv2.contourArea(c) > 1000]
                
                if contornos_validos:
                    cv2.drawContours(frame_bgr, contornos_validos, -1, (0, 0, 255), 2)
                    
                    dist_zonas = {"izq": float('inf'), "cen": float('inf'), "der": float('inf')}
                    w_tercio = w_img // 3
                    
                    # Evaluar todos los puntos del contorno para profundidad
                    for c in contornos_validos:
                        for punto in c:
                            px_x, px_y = punto[0]
                            px_x = max(0, min(w_img - 1, px_x))
                            px_y = max(0, min(h_img - 1, px_y))
                            
                            z = pc_data[px_y, px_x, 2]
                            
                            if np.isfinite(z) and z > 0:
                                if px_x < w_tercio:
                                    dist_zonas["izq"] = min(dist_zonas["izq"], z)
                                elif px_x < 2 * w_tercio:
                                    dist_zonas["cen"] = min(dist_zonas["cen"], z)
                                else:
                                    dist_zonas["der"] = min(dist_zonas["der"], z)
                    
                    # Preparar distancias
                    d_izq = dist_zonas["izq"] if dist_zonas["izq"] != float('inf') else 0.0
                    d_cen = dist_zonas["cen"] if dist_zonas["cen"] != float('inf') else 0.0
                    d_der = dist_zonas["der"] if dist_zonas["der"] != float('inf') else 0.0
                    
                    # Dibujar UI
                    cv2.line(frame_bgr, (w_tercio, 0), (w_tercio, h_img), (255, 255, 0), 1)
                    cv2.line(frame_bgr, (2 * w_tercio, 0), (2 * w_tercio, h_img), (255, 255, 0), 1)
                    cv2.putText(frame_bgr, f"IZQ: {d_izq:.2f}m", (10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f"CEN: {d_cen:.2f}m", (w_tercio + 10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f"DER: {d_der:.2f}m", (2 * w_tercio + 10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            if ret:
                latest_frame = buffer.tobytes()
        time.sleep(0.01)

def generar_video():
    global latest_frame
    while True:
        if latest_frame is not None:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.05)

@app.route('/video')
def video_feed():
    return Response(generar_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask():
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)

# ==========================================
# RUTINAS DE CONTROL SERIAL
# ==========================================
def enviar_movimiento(m1, m2, m3, m4):
    if ser is None or not ser.is_open:
        return
        
    m1_final = m1 * -1
    m2_final = m2       
    m3_final = m3       
    m4_final = m4 * -1
    
    trama_completa = [m1_final, m2_final, m3_final, m4_final] + estado_pca
    comando = ",".join(map(str, trama_completa)) + "\n"
    ser.write(comando.encode('utf-8'))

def mover_robot(direccion, milisegundos):
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

    while time.time() - start_time < duracion_segundos:
        enviar_movimiento(izq, izq, der, der)
        time.sleep(0.05)
    
    enviar_movimiento(0, 0, 0, 0)

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == '__main__':
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=start_flask, daemon=True).start()
    print("[INFO] Visión y servidor Flask en segundo plano -> http://<IP_ROBOT>:5000/video")

    try:
        ser = serial.Serial(puerto, baudios, timeout=0.1)
        time.sleep(2) 
        ser.reset_input_buffer()
        print(f"[INFO] Conectado a {puerto}. PWM configurado a {PWM_MAX}.")
        print("\nComandos: [w/a/s/d] [ms] | p [pin_pca] [pwm] | Ctrl+C para salir\n")

        while True:
            entrada = input(">> ").strip().lower().split()
            if not entrada: continue
            comando = entrada[0]

            if comando in ['w', 'a', 's', 'd'] and len(entrada) == 2:
                try: mover_robot(comando, int(entrada[1]))
                except ValueError: print("Error: ms debe ser entero.")
            elif comando == 'p' and len(entrada) == 3:
                try:
                    pin, val = int(entrada[1]), int(entrada[2])
                    if 0 <= pin <= 15 and -1 <= val <= 4095:
                        estado_pca[pin] = val
                        enviar_movimiento(0, 0, 0, 0)
                        print(f"PCA Pin {pin} -> {val}")
                    else: print("Error: Pin(0-15), PWM(-1-4095)")
                except ValueError: print("Error: Valores deben ser enteros.")
            else: print("Comando no reconocido.")

    except serial.SerialException as e:
        print(f"Error puerto serial: {e}")
    except KeyboardInterrupt:
        print("\nSaliendo...")
    finally:
        if ser and ser.is_open:
            try:
                for _ in range(5):
                    ser.write(("0,0,0,0," + ",".join(["-1"] * 16) + "\n").encode('utf-8'))
                    time.sleep(0.01)
            except: pass 
            ser.close()
        print("Detenido.")
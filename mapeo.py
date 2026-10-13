import sys
import cv2
import numpy as np
import pyzed.sl as sl
import threading
from flask import Flask, Response

# ==========================================
# 0. VARIABLES GLOBALES
# ==========================================
ESTADO = "ESPERANDO" # ESPERANDO -> MAPEANDO -> NAVEGANDO
keep_running = True

trayectoria_mapeo = []
limites_mapa = {'x_min': 0.0, 'x_max': 0.0, 'z_min': 0.0, 'z_max': 0.0}

# ==========================================
# 1. HILO PARA LEER LA TERMINAL
# ==========================================
def leer_terminal():
    global ESTADO
    print("\n" + "="*50)
    print("CONTROLES DEL MAPEO (Escribe aquí y presiona Enter):")
    print("  'I' -> INICIAR mapeo")
    print("  'T' -> TERMINAR mapeo y empezar a navegar")
    print("="*50 + "\n")
    
    while keep_running:
        try:
            comando = input().strip().upper()
            if comando == 'I' and ESTADO == "ESPERANDO":
                print("\n[ACCION] Iniciando Mapeo... Mueve la cámara para delimitar el área.")
                ESTADO = "MAPEANDO"
            elif comando == 'T' and ESTADO == "MAPEANDO":
                print("\n[ACCION] Terminando Mapeo. Calculando límites seguros...")
                ESTADO = "NAVEGANDO"
            else:
                print(f"Comando '{comando}' no válido o fuera de tiempo (Estado actual: {ESTADO})")
        except EOFError:
            break

# ==========================================
# 2. CONFIGURACIÓN ZED Y FLASK
# ==========================================
app = Flask(__name__)

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Suficiente para Tracking

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("No se pudo abrir la cámara ZED.")
    sys.exit(1)

# Habilitar Tracking Posicional
tracking_params = sl.PositionalTrackingParameters()
zed.enable_positional_tracking(tracking_params)

def generar_minimapa():
    global ESTADO, limites_mapa, trayectoria_mapeo
    
    pose = sl.Pose()
    runtime_parameters = sl.RuntimeParameters()
    
    # Parámetros visuales del minimapa 2D (Fondo negro)
    ancho_img, alto_img = 600, 600
    centro_x, centro_z = ancho_img // 2, alto_img // 2
    escala = 50 # 50 pixeles equivalen a 1 metro en la vida real

    while keep_running:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Obtener posición exacta
            estado_tracking = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
            
            if estado_tracking == sl.POSITIONAL_TRACKING_STATE.OK:
                # Extraemos X (lateral) y Z (profundidad). Ignoramos Y (altura).
                translacion = pose.get_translation().get()
                pos_x = translacion[0]
                pos_z = translacion[2]
                
                minimapa = np.zeros((alto_img, ancho_img, 3), dtype=np.uint8)
                
                # Lógica de Mapeo
                if ESTADO == "MAPEANDO":
                    trayectoria_mapeo.append((pos_x, pos_z))
                    # Actualizar caja de límites extremos
                    limites_mapa['x_min'] = min([p[0] for p in trayectoria_mapeo])
                    limites_mapa['x_max'] = max([p[0] for p in trayectoria_mapeo])
                    limites_mapa['z_min'] = min([p[1] for p in trayectoria_mapeo])
                    limites_mapa['z_max'] = max([p[1] for p in trayectoria_mapeo])
                
                # Lógica Visual (Dibujar mapa y posición)
                color_estado = (255, 255, 255) # Blanco por defecto
                
                if ESTADO in ["MAPEANDO", "NAVEGANDO"] and len(trayectoria_mapeo) > 0:
                    px_min = int(centro_x + limites_mapa['x_min'] * escala)
                    px_max = int(centro_x + limites_mapa['x_max'] * escala)
                    pz_min = int(centro_z + limites_mapa['z_min'] * escala)
                    pz_max = int(centro_z + limites_mapa['z_max'] * escala)
                    
                    # Dibujar el rectángulo de la zona mapeada
                    cv2.rectangle(minimapa, (px_min, pz_min), (px_max, pz_max), (100, 100, 100), 2)
                    
                    if ESTADO == "NAVEGANDO":
                        # Comprobar "Geofence" (¿Estoy dentro o fuera?)
                        dentro_x = limites_mapa['x_min'] <= pos_x <= limites_mapa['x_max']
                        dentro_z = limites_mapa['z_min'] <= pos_z <= limites_mapa['z_max']
                        
                        if dentro_x and dentro_z:
                            cv2.putText(minimapa, "DENTRO DEL MAPA", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            color_estado = (0, 255, 0)
                        else:
                            cv2.putText(minimapa, "! FUERA DEL MAPA !", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            color_estado = (0, 0, 255)

                # Dibujar el "Robot/Cámara"
                coord_pixel_x = int(centro_x + pos_x * escala)
                coord_pixel_z = int(centro_z + pos_z * escala)
                cv2.circle(minimapa, (coord_pixel_x, coord_pixel_z), 8, color_estado, -1)

                # HUD Info
                cv2.putText(minimapa, f"ESTADO: {ESTADO}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(minimapa, f"X: {pos_x:.2f}m  Z: {pos_z:.2f}m", (20, alto_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                # Enviar al navegador
                ret, buffer = cv2.imencode('.jpg', minimapa)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generar_minimapa(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Iniciar hilo para leer el teclado
    hilo_teclado = threading.Thread(target=leer_terminal, daemon=True)
    hilo_teclado.start()
    
    print("\n[INFO] Servidor listo. Abre http://192.168.55.1:5000 en tu navegador.")
    try:
        # Flask usa la terminal principal para correr el servidor web
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\nCerrando sistema...")
        keep_running = False
        zed.disable_positional_tracking()
        zed.close()
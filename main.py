import sys
import signal
import time
import serial
import logging as log

from vision import VisionSystem
from serial_read import leer_esp32
from serial_write import escribir_esp32
from control import NavegacionSMC

log.basicConfig(level=log.INFO)

PUERTO_SERIAL = '/dev/ttyUSB0'
BAUDIOS = 115200
try:
    ser = serial.Serial(PUERTO_SERIAL, BAUDIOS, timeout=0.1)
    time.sleep(2)
    ser.reset_input_buffer()
    log.info("Puerto serial abierto correctamente.")
except serial.SerialException as e:
    log.error(f"Error al abrir el puerto serial: {e}")
    ser = None

log.info("Iniciando sistema de visión y servidor Flask...")
vision_sys = VisionSystem()
vision_sys.start_server()

log.info("Inicializando Controlador SMC...")
controlador = NavegacionSMC()

# Inicializamos encoders
enc1 = enc2 = enc3 = enc4 = 0.0

SERVO_MIN_PWM = 100

def apagar_sistema(sig, frame):
    print("\n[INFO] Apagando el sistema por completo...")
    vision_sys.stop()
    if ser and ser.is_open:
        escribir_esp32(ser, 0, 0, 0, 0, [-1]*16)
        time.sleep(0.1)
        ser.close()
    sys.exit(0)

signal.signal(signal.SIGINT, apagar_sistema)

if __name__ == "__main__":
    log.info("Sistema en ejecución. Presiona Ctrl+C para detener.")

    if ser and ser.is_open:
        estado_pca_inicial = [-1] * 16
        estado_pca_inicial[3] = SERVO_MIN_PWM
        estado_pca_inicial[5] = SERVO_MIN_PWM
        escribir_esp32(ser, 0, 0, 0, 0, estado_pca_inicial)
        log.info("Servos 3 y 5 movidos a posición inicial.")
        time.sleep(0.5)

    try:
        while True:
            # A. Visión
            (lata_detectada, obj_x, obj_z, color,
             d_borde_izq, d_borde_cen, d_borde_der) = vision_sys.get_data()

            # B. Encoders
            datos_esp32 = leer_esp32(ser)
            if datos_esp32:
                try:
                    partes = datos_esp32.split(',')
                    if len(partes) >= 4:
                        enc1, enc2, enc3, enc4 = map(float, partes[:4])
                except ValueError:
                    pass

            # C. Control
            m1, m2, m3, m4 = controlador.calcular_control(
                lata_detectada, obj_x, obj_z,
                enc1, enc2, enc3, enc4,
                d_borde_izq, d_borde_cen, d_borde_der
            )

            # D. Actuador siempre encendido
            pca = controlador.calcular_pca()

            # E. Enviar
            if ser and ser.is_open:
                escribir_esp32(ser, m1, m2, m3, m4, pca)

            # F. Print Telemetría Completa
            str_estado = controlador._nombre_estado(controlador.estado)
            str_coords = f"X:{obj_x:.2f} Z:{obj_z:.2f}" if (lata_detectada and obj_x is not None and obj_z is not None) else "---"
            
            # Formato en una sola línea compacta pero con todos los datos
            print(f"Est: {str_estado:<12} | Tht: {controlador.theta_robot:+.2f} | "
                  f"Lat: [{str_coords:<13}] | Bord(I,C,D): {d_borde_izq:4.1f},{d_borde_cen:4.1f},{d_borde_der:4.1f} | "
                  f"Enc: {enc1:5.1f},{enc2:5.1f},{enc3:5.1f},{enc4:5.1f} | "
                  f"Atasco: {controlador._ciclos_atascado:2d}/50 | "
                  f"PWM: [{m1:4d},{m2:4d},{m3:4d},{m4:4d}]")

            time.sleep(0.05)

    except Exception as e:
        log.error(f"Error fatal en el bucle principal: {e}")
        apagar_sistema(None, None)
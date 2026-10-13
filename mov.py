import serial
import time
import sys

# --- Configuración General ---
puerto = '/dev/ttyUSB0' # Ajustar si es necesario
baudios = 115200

# Variable para ajustar la velocidad máxima de los movimientos (0 a 255)
PWM_MAX = 150

# --- Configuración de Servos (PCA9685 a 50Hz) ---
# Ajusta estos valores según las especificaciones de tus servos físicos
SERVO_MIN_PWM = 100  # Pulso para 0 grados (~0.5ms - 1ms)
SERVO_MAX_PWM = 500  # Pulso para 180 grados (~2ms - 2.5ms)

# Estado por defecto de los 16 canales del PCA9685 (-1 = ignorar)
estado_pca = [-1] * 16

# Inicializar la variable del puerto vacía
ser = None

def enviar_movimiento(m1, m2, m3, m4):
    """Aplica la corrección física de los motores invertidos y envía a la ESP32 la trama completa."""
    if ser is None or not ser.is_open:
        return
        
    m1_final = m1 * -1
    m2_final = m2       
    m3_final = m3       
    m4_final = m4 * -1
    
    # Concatenamos los 4 motores con los 16 estados del PCA
    trama_completa = [m1_final, m2_final, m3_final, m4_final] + estado_pca
    comando = ",".join(map(str, trama_completa)) + "\n"
    
    ser.write(comando.encode('utf-8'))

def mover_robot(direccion, milisegundos):
    """Calcula las velocidades tipo tanque y mantiene el movimiento el tiempo indicado."""
    # Lógica de tracción diferencial (tanque)
    if direccion == 'w':   # Adelante
        izq, der = PWM_MAX, PWM_MAX
    elif direccion == 's': # Atrás
        izq, der = -PWM_MAX, -PWM_MAX
    elif direccion == 'a': # Izquierda (Rotación)
        izq, der = -PWM_MAX, PWM_MAX
    elif direccion == 'd': # Derecha (Rotación)
        izq, der = PWM_MAX, -PWM_MAX
    else:
        print("Dirección no válida.")
        return

    duracion_segundos = milisegundos / 1000.0
    start_time = time.time()

    # Bucle para mantener vivo el failsafe del ESP32 durante el tiempo estipulado
    while time.time() - start_time < duracion_segundos:
        # Lado IZQUIERDO = M1 y M2 | Lado DERECHO = M3 y M4
        enviar_movimiento(izq, izq, der, der)
        time.sleep(0.05) # ~20Hz
    
    # Detener motores al terminar el tiempo
    enviar_movimiento(0, 0, 0, 0)

# --- Bucle de Control Principal ---
try:
    ser = serial.Serial(puerto, baudios, timeout=0.1)
    time.sleep(2) 
    ser.reset_input_buffer()
    print(f"Conectado a {puerto}. Velocidad configurada a {PWM_MAX} PWM.")

    print("\n--- Control del Robot y Actuadores ---")
    print("Movimiento: [w/a/s/d] [milisegundos] (Ejemplo: w 1500)")
    print("Motor DC (Pin 6): p 6 [PWM]          (Ejemplo: p 6 2047 | Rango: 0 a 4095)")
    print("Servos (Otros):   p [pin] [grados]   (Ejemplo: p 0 90   | Rango: 0 a 180)")
    print("Servos simultáneos: p35 [grados]     (Ejemplo: p35 90   | Mueve pines 3 y 5 a la vez)")
    print("                  (-1 para ignorar cualquier actuador)")
    print("Presiona Ctrl+C para salir.\n")

    while True:
        entrada = input("Comando >> ").strip().lower().split()
        
        if len(entrada) == 0:
            continue
            
        comando = entrada[0]

        # 1. Rutina para Mover el Robot
        if comando in ['w', 'a', 's', 'd']:
            if len(entrada) == 2:
                try:
                    tiempo_ms = int(entrada[1])
                    mover_robot(comando, tiempo_ms)
                except ValueError:
                    print("Error: Los milisegundos deben ser un número entero.")
            else:
                print("Error: Formato de movimiento incorrecto. Usa: [tecla] [milisegundos]")

        # 2. Rutina para Controlar Actuadores (PCA9685)
        elif comando == 'p':
            if len(entrada) == 3:
                try:
                    pin_pca = int(entrada[1])
                    valor = int(entrada[2])

                    if 0 <= pin_pca <= 15:
                        
                        # --- LÓGICA PARA EL MOTOR DC EN EL PIN 6 ---
                        if pin_pca == 6:
                            if -1 <= valor <= 4095:
                                estado_pca[pin_pca] = valor
                                enviar_movimiento(0, 0, 0, 0)
                                print(f"Motor DC actualizado -> Pin: {pin_pca}, PWM: {valor}")
                            else:
                                print("Error: Para el motor (pin 6), el PWM debe estar entre 0 y 4095 (-1 para ignorar).")
                        
                        # --- LÓGICA PARA LOS SERVOS EN LOS DEMÁS PINES ---
                        else:
                            if valor == -1:
                                estado_pca[pin_pca] = -1
                                enviar_movimiento(0, 0, 0, 0)
                                print(f"Servo ignorado -> Pin: {pin_pca}")
                            elif 0 <= valor <= 180:
                                # Mapeo matemático de 0-180 grados a los límites de PWM del servo
                                valor_pwm = int((valor / 180.0) * (SERVO_MAX_PWM - SERVO_MIN_PWM) + SERVO_MIN_PWM)
                                estado_pca[pin_pca] = valor_pwm
                                enviar_movimiento(0, 0, 0, 0)
                                print(f"Servo actualizado -> Pin: {pin_pca}, Posición: {valor}° (PWM interno: {valor_pwm})")
                            else:
                                print("Error: Para los servos, la posición debe estar entre 0 y 180 grados (-1 para ignorar).")
                    else:
                        print("Error: El pin del PCA debe estar entre 0 y 15.")
                except ValueError:
                    print("Error: El pin y el valor deben ser números enteros.")
            else:
                print("Error: Formato de actuador incorrecto. Usa: p [pin] [valor]")
        
        # 3. Rutina para mover los pines 3 y 5 al mismo tiempo
        elif comando == 'p35':
            if len(entrada) == 2:
                try:
                    valor = int(entrada[1])
                    if valor == -1:
                        estado_pca[3] = -1
                        estado_pca[5] = -1
                        enviar_movimiento(0, 0, 0, 0)
                        print("Servos 3 y 5 ignorados.")
                    elif 0 <= valor <= 180:
                        valor_pwm = int((valor / 180.0) * (SERVO_MAX_PWM - SERVO_MIN_PWM) + SERVO_MIN_PWM)
                        estado_pca[3] = valor_pwm
                        estado_pca[5] = valor_pwm
                        enviar_movimiento(0, 0, 0, 0)
                        print(f"Servos 3 y 5 actualizados simultáneamente -> Posición: {valor}° (PWM interno: {valor_pwm})")
                    else:
                        print("Error: La posición debe estar entre 0 y 180 grados (-1 para ignorar).")
                except ValueError:
                    print("Error: El valor debe ser un número entero.")
            else:
                print("Error: Formato incorrecto. Usa: p35 [grados]")

        else:
            print("Error: Comando no reconocido.")

except serial.SerialException as e:
    print(f"Error al abrir o comunicar con el puerto: {e}")
except KeyboardInterrupt:
    print("\nDeteniendo programa...")
finally:
    # Asegura el cierre sin importar dónde falló el código
    if ser is not None and ser.is_open:
        try:
            # Parada segura de emergencia al cerrar
            trama_parada = "0,0,0,0," + ",".join(["-1"] * 16) + "\n"
            for _ in range(5):
                ser.write(trama_parada.encode('utf-8'))
                time.sleep(0.01)
        except Exception:
            pass # Ignora errores de escritura durante el cierre
        
        ser.close()
        print("Puerto cerrado correctamente. Robot detenido.")
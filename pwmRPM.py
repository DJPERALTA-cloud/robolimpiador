import serial
import time

# ==========================================
# --- CONFIGURACIÓN DE LA PRUEBA PWM-RPM ---
# ==========================================
puerto = '/dev/ttyUSB0' # Asegúrate de usar tu puerto correcto
baudios = 115200

# Parámetros del experimento
PWM_INICIO = 70 
PWM_FINAL = 255        
SALTO_PWM = 25        
TIEMPO_DURACION_SEG = 0.5 # Tiempo total para sumar iteraciones

# Estado por defecto de los 16 canales del PCA9685
estado_pca = [-1] * 16
ser = None

def enviar_movimiento(m1, m2, m3, m4):
    """Aplica la corrección física y envía los comandos a la ESP32."""
    if ser is None or not ser.is_open:
        return
        
    trama_completa = [m1 * -1, m2, m3, m4 * -1] + estado_pca
    comando = ",".join(map(str, trama_completa)) + "\n"
    
    try:
        ser.write(comando.encode('utf-8'))
    except Exception:
        pass

def ejecutar_prueba():
    print("PWM_Aplicado,RPM_M1,RPM_M2,RPM_M3,RPM_M4", flush=True)
    
    rango_pwm = list(range(PWM_INICIO, PWM_FINAL + SALTO_PWM, SALTO_PWM))
    rango_pwm = [pwm for pwm in rango_pwm if pwm <= PWM_FINAL]

    for pwm_actual in rango_pwm:
        # 1. Enviar el PWM actual a los motores
        enviar_movimiento(pwm_actual, pwm_actual, pwm_actual, pwm_actual)
        
        # 2. Variables para sumar TODO durante el paso
        suma_m1 = suma_m2 = suma_m3 = suma_m4 = 0.0
        contador_iteraciones = 0
        
        inicio_paso = time.time()
        
        # 3. Recolectar e iterar por la duración total del escalón (ej. 3 segundos)
        while time.time() - inicio_paso < TIEMPO_DURACION_SEG:
            if ser.in_waiting > 0:
                try:
                    linea = ser.readline().decode('utf-8', errors='ignore').strip()
                    if linea:
                        partes = linea.split(',')
                        if len(partes) == 4:
                            # Sumar iteraciones a sus respectivos totales
                            suma_m1 += float(partes[0])
                            suma_m2 += float(partes[1])
                            suma_m3 += float(partes[2])
                            suma_m4 += float(partes[3])
                            contador_iteraciones += 1
                except ValueError:
                    pass
            else:
                time.sleep(0.005) # Pausa mínima para no saturar la CPU
        
        # 4. Termina el tiempo del escalón: Calcular el promedio global
        if contador_iteraciones > 0:
            promedio_m1 = suma_m1 / contador_iteraciones
            promedio_m2 = suma_m2 / contador_iteraciones
            promedio_m3 = suma_m3 / contador_iteraciones
            promedio_m4 = suma_m4 / contador_iteraciones
            
            print(f"{pwm_actual},{promedio_m1:.2f},{promedio_m2:.2f},{promedio_m3:.2f},{promedio_m4:.2f}", flush=True)
        else:
            print(f"{pwm_actual},0.00,0.00,0.00,0.00", flush=True)
            
    # Detener todo
    enviar_movimiento(0, 0, 0, 0)
    print("\n--- PRUEBA FINALIZADA ---", flush=True)

# ==========================================
# --- BUCLE PRINCIPAL DE EJECUCIÓN ---
# ==========================================
try:
    ser = serial.Serial(puerto, baudios, timeout=0.1)
    time.sleep(2) 
    ser.reset_input_buffer()
    
    ejecutar_prueba()

except serial.SerialException as e:
    print(f"Error con el puerto: {e}")
except KeyboardInterrupt:
    print("\nDeteniendo...", flush=True)
finally:
    if ser is not None and ser.is_open:
        try:
            trama_parada = "0,0,0,0," + ",".join(["-1"] * 16) + "\n"
            for _ in range(5):
                ser.write(trama_parada.encode('utf-8'))
                time.sleep(0.01)
        except Exception:
            pass 
        ser.close()
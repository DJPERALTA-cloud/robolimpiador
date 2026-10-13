import serial
import time

# --- Configuración General ---
puerto = '/dev/ttyUSB0' 
baudios = 115200
PWM_MAX = 254
estado_pca = [-1] * 16
PASO_PWM = 15  # Incremento del PWM en cada iteración
TIEMPO_ESTABILIZACION = 0.150 # Segundos entre cada incremento

def enviar_movimiento(ser, m1, m2, m3, m4):
    """Aplica la corrección física de los motores invertidos y envía a la ESP32."""
    if ser is None or not ser.is_open:
        return
        
    m1_final = m1 * -1
    m2_final = m2       
    m3_final = m3       
    m4_final = m4 * -1
    
    trama_completa = [m1_final, m2_final, m3_final, m4_final] + estado_pca
    comando = ",".join(map(str, trama_completa)) + "\n"
    
    ser.write(comando.encode('utf-8'))

def leer_encoders(ser):
    """Lee el puerto serie y extrae la última trama válida de encoders."""
    datos_validos = None
    # Vaciar el buffer leyendo todas las líneas disponibles y quedarnos con la última
    while ser.in_waiting > 0:
        try:
            linea = ser.readline().decode('utf-8').strip()
            if linea:
                partes = linea.split(',')
                if len(partes) >= 4:
                    datos_validos = {
                        'TD': float(partes[0]), # Trasero derecho
                        'DD': float(partes[1]), # Delantero derecho
                        'DI': float(partes[2]), # Delantero izquierdo
                        'TI': float(partes[3])  # Trasero izquierdo
                    }
        except (ValueError, UnicodeDecodeError):
            pass # Ignorar basura o texto de debug
            
    return datos_validos

# --- Bucle Principal ---
ser = None
try:
    ser = serial.Serial(puerto, baudios, timeout=0.1)
    time.sleep(2) # Tiempo para que la ESP32 reinicie si es necesario
    ser.reset_input_buffer()
    
    print(f"Conectado a {puerto}. Iniciando rampa de PWM...")
    
    # Bucle de incremento de PWM
    for pwm in range(0, PWM_MAX + 1, PASO_PWM):
        # Enviar PWM (movimiento hacia adelante 'w')
        enviar_movimiento(ser, pwm, pwm, pwm, pwm)
        
        # Esperar a que el motor reaccione y la ESP32 envíe datos
        time.sleep(TIEMPO_ESTABILIZACION)
        
        # Leer retroalimentación
        encoders = leer_encoders(ser)
        
        if encoders:
            print(f"PWM Aplicado: {pwm:>3} | Encoders (RPM) -> DI: {encoders['DI']:>6.2f} | TI: {encoders['TI']:>6.2f} | DD: {encoders['DD']:>6.2f} | TD: {encoders['TD']:>6.2f}")
        else:
            print(f"PWM Aplicado: {pwm:>3} | Encoders (RPM) -> Sin datos recibidos")

    print("\nRampa finalizada.")

except serial.SerialException as e:
    print(f"Error de puerto serie: {e}")
except KeyboardInterrupt:
    print("\nPrueba interrumpida manualmente.")
finally:
    # Parada segura al terminar o fallar
    if ser is not None and ser.is_open:
        try:
            enviar_movimiento(ser, 0, 0, 0, 0)
            ser.close()
            print("Motores detenidos. Puerto cerrado.")
        except Exception as e:
            print(f"Error al cerrar: {e}")
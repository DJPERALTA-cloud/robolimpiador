import serial
import time

puerto = '/dev/ttyUSB0' # Actualiza según lo encontrado en el Paso 2
baudios = 115200

try:
    # Configurar puerto serial
    ser = serial.Serial(puerto, baudios, timeout=1)
    
    # IMPORTANTE: El ESP32 se reinicia físicamente al abrir una conexión serial.
    # Necesitas esperar 2 segundos antes de enviar datos para no perderlos.
    time.sleep(2) 

except serial.SerialException as e:
    print(f"Error al abrir el puerto: {e}")
    exit()

def enviar_comando(comando):
    # Agregar salto de línea y codificar a bytes (UTF-8)
    mensaje = comando + '\n'
    ser.write(mensaje.encode('utf-8'))
    
    # Leer confirmación del ESP32
    if ser.in_waiting > 0:
        respuesta = ser.readline().decode('utf-8').strip()
        print(respuesta)

# Ejecución de prueba
print("Enviando orden de ENCENDER...")
enviar_comando("ENCENDER")

time.sleep(3) # Mantener encendido 3 segundos

print("Enviando orden de APAGAR...")
enviar_comando("APAGAR")

# Cerrar puerto de forma segura
ser.close()
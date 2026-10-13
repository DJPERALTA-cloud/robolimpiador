import serial
import time

# Usa el mismo formato de puerto que te funcionó en el otro script (ttyUSB0 o ttyUSB1)
PUERTO = '/dev/ttyUSB0' 
BAUDRATE = 115200

def leer_ultrasonicos():
    ser = None
    try:
        # Timeout a 0.5 para dar holgura a las lecturas de 100ms del ESP32
        ser = serial.Serial(PUERTO, BAUDRATE, timeout=0.5)
        
        # Evita el reinicio en bucle del ESP32
        ser.setDTR(False)
        time.sleep(2) 
        ser.reset_input_buffer()
        print(f"Conectado a {PUERTO} a {BAUDRATE} bps.")
        print("Buffer limpio. Recibiendo datos de los sensores...\n")

        while True:
            if ser.in_waiting > 0:
                linea_bytes = ser.readline()
                
                if linea_bytes:
                    linea = linea_bytes.decode('utf-8', errors='ignore').strip()
                    
                    if linea:
                        datos = linea.split(',')
                        
                        if len(datos) == 2:
                            try:
                                sensor1 = float(datos[0])
                                sensor2 = float(datos[1])
                                print(f"Sensor 1: {sensor1:.3f} m  |  Sensor 2: {sensor2:.3f} m")
                            except ValueError:
                                pass

    except serial.SerialException as e:
        print(f"Error al abrir o comunicar con el puerto: {e}")
    except KeyboardInterrupt:
        print("\nDeteniendo programa...")
    finally:
        if ser is not None and ser.is_open:
            ser.close()
            print("Puerto cerrado correctamente.")

if __name__ == '__main__':
    leer_ultrasonicos()
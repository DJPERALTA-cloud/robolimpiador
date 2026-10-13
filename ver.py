import time
from vision import VisionSystem

if __name__ == "__main__":
    print("Llamando al módulo de visión...")
    
    # Instanciamos la clase importada del otro archivo
    mi_vision = VisionSystem()
    
    # Iniciamos el servidor Flask en el puerto 5000
    mi_vision.start_server()
    
    print("=========================================================")
    print("Cámara y servidor iniciados.")
    print("Visualiza el entorno en: http://localhost:5000/video")
    print("Presiona Ctrl+C en esta terminal para detener.")
    print("=========================================================")

    try:
        # Mantenemos el script principal vivo para que los hilos sigan trabajando
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nDeteniendo el sistema...")
    finally:
        # Aseguramos que la cámara se cierre correctamente
        mi_vision.stop()
        print("Sistema cerrado.")
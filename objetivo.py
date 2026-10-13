import time
from vision import VisionSystem

if __name__ == "__main__":
    print("Llamando al módulo de visión...")
    
    # Instanciamos la clase importada
    mi_vision = VisionSystem()
    
    # Iniciamos el servidor Flask
    mi_vision.start_server()
    
    print("=========================================================")
    print("Cámara y servidor iniciados.")
    print("Visualiza el entorno en: http://localhost:5000/video")
    print("Presiona Ctrl+C en esta terminal para detener.")
    print("=========================================================")

    try:
        while True:
            # Mandamos llamar exclusivamente a la nueva función
            detectado, obj_x, obj_z = mi_vision.get_target_data()
            
            if detectado:
                # Calculamos la distancia euclidiana total
                distancia = (obj_x**2 + obj_z**2)**0.5
                print(f"[OBJETIVO ENCONTRADO] -> Posición X: {obj_x:.2f}m | Profundidad Z: {obj_z:.2f}m | Distancia Directa: {distancia:.2f}m")
            else:
                print("[BUSCANDO] No se detecta el objetivo rojo con las características requeridas.")
                
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nDeteniendo el sistema...")
    finally:
        mi_vision.stop()
        print("Sistema cerrado.")
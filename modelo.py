from ultralytics import YOLO

# 1. Cargar tu modelo original
print("Cargando modelo PyTorch original...")
model = YOLO("best.pt")

# 2. Exportar a TensorRT (formato .engine)
print("Iniciando exportación a TensorRT. Esto puede tardar varios minutos...")
export_path = model.export(
    format="engine",  # Formato TensorRT
    device=0,         # Usar la GPU
    half=True,        # Activar FP16 para máxima velocidad en la Jetson
    imgsz=640,        # Tamaño de imagen de entrada (ajusta si usaste otro al entrenar)
    workspace=4       # Memoria máxima en GB permitida para la conversión (4 es seguro)
)

print(f"\n¡Exportación completada exitosamente!")
print(f"Tu nuevo modelo ultra-rápido está aquí: {export_path}")
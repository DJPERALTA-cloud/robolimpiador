import pyzed.sl as sl
from flask import Flask, Response
import cv2
import numpy as np
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Comprobación directa de CUDA antes de iniciar
if torch.cuda.is_available():
    print("PyTorch detectó CUDA correctamente. Usando GPU.")
else:
    print("ADVERTENCIA: PyTorch NO detecta CUDA. Verifica la instalación de PyTorch para Jetson.")

model = YOLO("best.pt", task='obb')

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA
init_params.camera_fps = 15
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Error al abrir la cámara ZED: {err}")
    exit(-1)

zed.enable_positional_tracking(sl.PositionalTrackingParameters())

obj_param = sl.ObjectDetectionParameters()
obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
obj_param.enable_tracking = True
zed.enable_object_detection(obj_param)

objects = sl.Objects()
obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

def generar_frames():
    mat = sl.Mat()
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            imagen_cv = mat.get_data() 
            imagen_bgr = cv2.cvtColor(imagen_cv, cv2.COLOR_BGRA2BGR)
            
            h, w = imagen_bgr.shape[:2]

            # CORRECCIÓN: device=0 fuerza la GPU, classes=[0] filtra solo el objeto 0
            results = model.predict(imagen_bgr, conf=0.2, imgsz=640, device=0, classes=[1], verbose=False)
            annotated_frame = results[0].plot()

            custom_objects = []
            
            for r in results:
                if r.obb is not None:
                    boxes = r.obb.xyxyxyxy.cpu().numpy()
                    confidences = r.obb.conf.cpu().numpy()
                    labels = r.obb.cls.cpu().numpy()

                    for box, conf, lbl in zip(boxes, confidences, labels):
                        # Limitar coordenadas entre 0 y los bordes
                        box[:, 0] = np.clip(box[:, 0], 0, w - 1)
                        box[:, 1] = np.clip(box[:, 1], 0, h - 1)
                        box_int = np.round(box).astype(int)

                        tmp = sl.CustomBoxObjectData()
                        tmp.bounding_box_2d = box_int
                        tmp.label = int(lbl)
                        tmp.probability = conf
                        tmp.is_grounded = False
                        custom_objects.append(tmp)

            # Ingestar solo si hay objetos de la clase 0 detectados
            if custom_objects:
                zed.ingest_custom_box_objects(custom_objects)
            
            zed.retrieve_objects(objects, obj_runtime_param)

            if objects.is_new:
                for obj in objects.object_list:
                    pos = obj.position
                    if not np.isnan(pos[2]) and len(obj.bounding_box_2d) > 0:
                        x, y, z = pos[0], pos[1], pos[2]
                        pt = (int(obj.bounding_box_2d[0][0]), int(obj.bounding_box_2d[0][1]) - 10)
                        texto_3d = f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}m"
                        cv2.putText(annotated_frame, texto_3d, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
                
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='192.168.55.1', port=5000)
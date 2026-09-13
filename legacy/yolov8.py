import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# UPLOAD MODEL
model = YOLO('yolov8n.pt') # Parece que es más rápido (unas 10 veces más, pero funciona peor)
#model = YOLO('yolov8x.pt')

# READ VIDEOPATH
video_path = r"C:\Users\ADMIN\Documents\xai_vd\datasets\Hockey fights\test\fight\fi5_xvid.avi"
# CREATE POINTER TO VIDEO
# Crea un puntero o manejador para el archivo de video sin almacenarlo en RAM. Esto es eficiente para no cargar el video en memoria pero poder trabajar con él
cap = cv2.VideoCapture(video_path)

# CREATE VARIABLES OUTSIDE FRAME LOOP
# Crea diccionario vacio para almacenar el historial de seguimiento de objetos
track_history = defaultdict(lambda: [])
# Lista de colores para asignar a cada persona
colors = [
    (255, 0, 0),   # Rojo
    (0, 255, 0),   # Verde
    (0, 0, 255),   # Azul
    (255, 255, 0), # Amarillo
    (0, 255, 255), # Cyan
    (255, 0, 255), # Magenta
    (255, 255, 255), # Blanco
    (0, 0, 0),     # Negro
    (128, 128, 128), # Gris
    (165, 42, 42),   # Marrón
    (255, 192, 203), # Rosa
    (138, 43, 226),  # Azul violeta
    (50, 205, 50),   # Verde lima
    (255, 215, 0),   # Oro
    (128, 0, 128),   # Púrpura
    (240, 230, 140), # Amarillo khaki
    (64, 224, 208),  # Turquesa
    (128, 0, 0),     # Marrón oscuro
    (139, 0, 139),   # Magenta oscuro
    (46, 139, 87)    # Verde mar
]
# Variable para hacer un seguimiento de los colores asignados
color_index = 0

# DEFINE VARIABLES FOR SAVING PROCESSED VIDEO
# Definir las propiedades del video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = r"C:\Users\ADMIN\Documents\xai_vd\try_videos\try.mp4"
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Crear el objeto VideoWriter
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        #YOLOV8 parameters: results = model.track(frame, imgsz=img_size, persist=True, conf=conf_level, tracker=tracker_option)
        results = model.track(frame, persist=True, classes=0)

        # Get the boxes and track IDs
        # .xywh indica que se están extrayendo las coordenadas de los cuadros delimitadores en formato (x, y, ancho, alto). X e Y son las de la esquina superior izquierda
        # Otra opción es utilizar xyxy.
        # Se utiliza para mover los datos desde la GPU (unidad de procesamiento gráfico) a la CPU (unidad de procesamiento central). Esto es necesario para trabajar con los datos en la CPU y realizar operaciones adicionales.
        #print(results[0].boxes.id)
        if results[0].boxes.id != None:
            boxes = results[0].boxes.xywh.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            #confidences = results[0].boxes.conf.cpu().numpy().astype(int)
            #class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        #track_ids = results[0].boxes.id.int().cpu().tolist()
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            #if len(track) > 30:  # retain 90 tracks for 90 frames
            #    track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
            color_index = (color_index + 1) % len(colors)  # Cycle through colors

        # Almacenar en el video a guardar el frame procesado
        out.write(annotated_frame)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        cv2.waitKey(0)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()

# Liberar el objeto VideoWriter
out.release()
cv2.destroyAllWindows()


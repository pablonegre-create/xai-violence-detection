import cv2
import numpy as np
import time

# Función para calcular la diferencia entre dos frames
def calcular_diferencia(frame_actual, frame_anterior):
    # Convertir los frames a escala de grises
    gray_actual = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
    gray_anterior = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)

    # Calcular la diferencia absoluta entre los frames
    diferencia = cv2.absdiff(gray_actual, gray_anterior)
    
    media = np.mean(diferencia)
    mediana = np.median(diferencia)
    desviacion_estandar = np.std(diferencia)
    minimo = np.min(diferencia)
    maximo = np.max(diferencia)

    print("Media:", media)
    print("Mediana:", mediana)
    print("Desviación estándar:", desviacion_estandar)
    print("Mínimo:", minimo)
    print("Máximo:", maximo)
    print("#######################")
    return diferencia

# Cargar el video
video = cv2.VideoCapture(r"c:\Users\ADMIN\Desktop\fi50_xvid.avi")

# Leer el primer frame
ret, frame_anterior = video.read()

# Verificar si el video se ha cargado correctamente
if not ret:
    print("Error al cargar el video")
    exit()

# Obtener la altura y anchura del video
altura, anchura, _ = frame_anterior.shape

# Configurar el codificador de video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_salida = cv2.VideoWriter(r"c:\Users\ADMIN\Desktop\dif_frames.mp4", fourcc, 20.0, (anchura, altura))

# Iterar a través de los frames del video
while True:
    # Leer el siguiente frame
    ret, frame_actual = video.read()

    # Salir del bucle si no hay más frames
    if not ret:
        break

    inicio = time.time()
    # Calcular la diferencia entre los frames
    diferencia = calcular_diferencia(frame_actual, frame_anterior)
    fin = time.time()
    dif_time = fin - inicio

    # Escribir el frame de diferencia en el video de salida
    video_salida.write(cv2.cvtColor(diferencia, cv2.COLOR_GRAY2BGR))

    # Mostrar el video de diferencia en tiempo real
    cv2.imshow('Diferencia de Frames', diferencia)
    
    # Actualizar el frame anterior con el frame actual para el siguiente ciclo
    frame_anterior = frame_actual

    # Detener el bucle al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
video.release()
video_salida.release()
cv2.destroyAllWindows()

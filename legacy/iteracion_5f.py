import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import load_model
from moviepy.editor import *


# X_train = np.load(r"C:\Users\ADMIN\Mi unidad (pablo.negre@usal.es)\Papers\Violence detection\results\hockey\no-finetunning\VGG19_extracted_features\train\train_X_lstm_input.npy")
# X_test = np.load(r"C:\Users\ADMIN\Mi unidad (pablo.negre@usal.es)\Papers\Violence detection\results\hockey\no-finetunning\VGG19_extracted_features\test\test_X_lstm_input.npy")
# target_frames = 40
# Cargo solo 1 video de cada uno 
#(quiero ver cual es el frame más relevante de un solo video)
# partial_X_train = X_train[:1, :, :]
# partial_X_test = X_train[:1, :, :]

def lstm_plot(df):
    plt.plot(df['Frame number'], df['Importance'])
    plt.xlabel('Frame Number')
    plt.ylabel('Importance')
    plt.show()

#def lstm_animation(predictions, output_gif, output_mp4):
    serie_temporal = predictions[:-5][["Importance"]]
    print(serie_temporal)
    fig, ax = plt.subplots()

    # Crear un gráfico vacío para la serie temporal (fondo negro)
    ax.plot(serie_temporal, color='black')
    plt.xlabel('Frame number')
    plt.ylabel('Frame relevance')

    # Crear la bola (será un punto en el gráfico)
    bola, = ax.plot([], [], marker='o', color='red')  # 'marker' define el marcador de la bola

    # Función de inicialización: se llama para crear la trama base
    def init():
        ax.set_xlim(0, len(serie_temporal))  # Establecer límites en x
        #ax.set_ylim(0, 1)  # Establecer límites en y
        return bola,

    # Función de animación: se llama para actualizar cada frame
    def animate(i):
        x = i  # Posición x para la bola (será el índice en la serie temporal)
        y = serie_temporal.iloc[i]  # Obtener el valor en el índice actual
        bola.set_data(x, y)  # Establecer los datos para la bola
        return bola,

    plt.tight_layout()

    # Crear la animación
    ani = animation.FuncAnimation(fig, animate, frames=len(serie_temporal), init_func=init,
                                interval=len(serie_temporal), blit=True)

    ani.save(output_gif, writer='ffmpeg', fps=5)
    
    # Convertir a mp4
    ruta_gif = r"C:\Users\ADMIN\Documents\xai_vd\try_videos\lstm_video.gif"
    ruta_video = r"C:\Users\ADMIN\Documents\xai_vd\try_videos\lstm_video.mp4"
    gif_clip = VideoFileClip(ruta_gif)
    gif_clip.write_videofile(ruta_video)
    os.remove(ruta_gif)
    
def xai_lstm_5f(feature_array, model, target_frames, output_gif, output_mp4, violence=True):
    
    # Definir el tamaño del paso (en este caso, 5)
    paso = 5
    # Crear una lista para almacenar los arrays después de eliminar los frames
    lista_predicciones = []
    # Iterar sobre los índices de los frames que se van a eliminar
    for i in range(0, 40, paso):
        # Obtener el índice final del rango a eliminar
        end_index = min(i + paso, 40)
        # Crear una nueva matriz sin los frames del rango actual
        nuevo_array = np.concatenate([feature_array[:, :i, :], feature_array[:, end_index:, :]], axis=1)
        # Agregar el último frame 5 veces
        ultimo_frame_repetido = np.repeat(nuevo_array[:, -1:, :], 5, axis=1)

        array_extendido = np.concatenate((nuevo_array, ultimo_frame_repetido), axis=1)
        # Predecir
        predicciones = model.predict(array_extendido)
        # Agregar la nueva matriz a la lista
        lista_predicciones.append(predicciones)

    lista_repetida = [elem for elem in lista_predicciones for _ in range(5)]
    original_predict = model.predict(feature_array)
    print("PREDICCION ORIGINAL: ")
    print(original_predict)
    original_no_violence_predict = original_predict[0][0]
    original_violence_predict = original_predict[0][1]
    predict = np.array(lista_repetida)
    predict = predict.reshape(target_frames, 2)
    df = pd.DataFrame(predict, columns=['no_v', 'y_v'])

    df['v_f_i'] = original_violence_predict/df['y_v']
    print(df)
    df['nv_f_i'] = original_no_violence_predict/df['no_v']
    df.reset_index(inplace=True)
    df['index'] = df['index'] + 1

"""    
    if violence:
        df.rename(columns={'v_f_i': 'Importance', 'index': 'Frame number'}, inplace=True)
        lstm_animation(df, output_gif, output_mp4)
    else:
        df.rename(columns={'no_f_i': 'Importance', 'index': 'Frame number'}, inplace=True)
        lstm_animation(df, output_gif, output_mp4)
"""
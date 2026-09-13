import os
import csv
import cv2
import datetime
import numpy as np
import pandas as pd

from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_tuner.tuners import RandomSearch
from keras.models import Model
from keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D


def preprocess_frame(frame):
    # Preprocesamiento de la imagen
    frame = cv2.resize(frame, (224, 224))
    return np.expand_dims(frame, axis=0)

def loop_each_video(video_path, target_frames):
    cap = cv2.VideoCapture(video_path)

    # Calcular el número total de fotogramas en el video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Truncar equidistantemente al número deseado de fotogramas
    indices_truncados = np.linspace(0, total_frames-1, target_frames, dtype=int)

    # Lista para almacenar las características de cada fotograma
    preprocessed_video = []

    # Iterar sobre cada fotograma
    for i in indices_truncados:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Solo seleccionara los frames seleccionados equidistantes
        ret, frame = cap.read()
        if not ret:
            break
        preprocessed_frame = preprocess_frame(frame)
        preprocessed_video.append(preprocessed_frame)

    preprocessed_video = np.array(preprocessed_video)
    # Liberar el objeto de captura
    cap.release()
    return preprocessed_video

def preprocess_testing_videos(video_dir, sequence_length):
    # Preprocesar videos de violencia y no violencia
    video_array = loop_each_video(video_dir, sequence_length)
    return video_array

def submodel_vgg19_global_average_pooling(vd_vgg19):
    # Create submodel: Input shape(None,224,224,3); Output shape: block5_pool(MaxPooling2D)(None, 7, 7, 512)
    output_layer = vd_vgg19.get_layer('block5_pool').output
    sub_vd_vgg19 = Model(inputs=vd_vgg19.input, outputs=output_layer)

    # Add Global Average Pooling Layer 2D to submodel
    output_tensor = sub_vd_vgg19.output
    output_tensor = GlobalAveragePooling2D()(output_tensor)
    sub_vd_vgg19_gap = Model(inputs=sub_vd_vgg19.input, outputs=output_tensor)
    return sub_vd_vgg19_gap

# MAIN

# PARAMETROS Y DIRECTORIOS
# video_dir = r"C:\Users\ADMIN\Documents\xai_vd\datasets\Hockey fights\test\fight\fi5_xvid.avi"
# sequence_length = 40  # Longitud de la secuencia temporal para la LSTM
# model_path = r"C:\Users\ADMIN\Documents\xai_vd\trained_models\02jan1334\biclass_vgg19.keras"
# vd_vgg19 = load_model(model_path)

def spatial_features(video_dir, sequence_length, vd_vgg19):
    video_array = preprocess_testing_videos(video_dir, sequence_length)

    # Reconvertir la dimension de la matriz para que sean todo secuencias de imagenes y no secuencias de imagenes de video
    X_test_reshape = np.reshape(video_array, (video_array.shape[0]*video_array.shape[1], 224, 224, 3))

    # Create submodel with globale average pooling layer
    sub_vd_vgg19_gap = submodel_vgg19_global_average_pooling(vd_vgg19)

    # Get predictions with output form (None, 512)
    predictions = sub_vd_vgg19_gap.predict(X_test_reshape)

    # Dejar las características extraídas con la forma que necesite la LSTM
    # Se debe pasar de (Nvideos*Nframes, 512) a (Nvideos, Nframes, 512)
    X_lstm_input = predictions.reshape((1, sequence_length, 512))
    return X_lstm_input


# Guardar array en directorio
# np.save(r"C:\Users\ADMIN\Documents\xai_vd\extracted_spatial_features\X_lstm_input", X_lstm_input)
# np.save(r"C:\Users\ADMIN\Documents\xai_vd\extracted_spatial_features\y_lstm_input", y_test)
# np.save(r"C:\Users\ADMIN\Documents\xai_vd\extracted_spatial_features\file_names", train_file_names)

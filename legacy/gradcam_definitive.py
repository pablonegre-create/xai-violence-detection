import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip
from PIL import Image
from tensorflow.keras.models import load_model

# Preprocesar video
def preprocess_frame(frame):
    # Preprocesamiento de la imagen
    frame = cv2.resize(frame, (224, 224))
    return np.expand_dims(frame, axis=0)

def preprocess_single_video(video_path, target_frames):

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
        preprocessed_frame = preprocess_frame(frame) # Forma (224,224,3)
        #preprocessed_frame = np.squeeze(preprocessed_frame) # Ahora de la forma (224,224,3)
        preprocessed_video.append(preprocessed_frame)

    # Liberar el objeto de captura
    cap.release()
    return preprocessed_video

# Apply gradcam to frame from some Conv. layer
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=1):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Heatmap
def loop_gradcam_heatmap(vgg_model, layer_name, video_array):
    all_heatmaps = []
    for conv_layer_i in layer_name:
        print("##############")
        print(conv_layer_i)
        
        nombre_video = r"C:\Users\ADMIN\Documents\xai_vd\try_videos"
        nombre_video = os.path.join(nombre_video, 'gradcam_heatmap_' + conv_layer_i + '.avi')
        fps = 5  # Fotogramas por segundo del video
        alto, ancho = 224, 224
        # Configurar el objeto VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(nombre_video, fourcc, fps, (ancho, alto))
        i = 0
        for frame in video_array:
            print(i)
            i = i+1
            # Grey image
            # gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
            # gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2RGB)

            # Get grad-cam heatmap
            heatmap = make_gradcam_heatmap(frame, vgg_model, conv_layer_i, pred_index=1)
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-9)
            heatmap = cv2.resize(heatmap, (224, 224))
            all_heatmaps.append(heatmap)
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_HOT)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            frame = frame.squeeze()
            imagen_superpuesta = cv2.addWeighted(frame, 0.3, heatmap, 0.7, 0)
            video.write(imagen_superpuesta)
        
        # Liberar recursos
        video.release()
        cv2.destroyAllWindows()
    all_heatmaps = np.array(all_heatmaps)
    new_shape = (5, 40, 224, 224)
    all_heatmaps = all_heatmaps.reshape(new_shape)
    mean_array = np.mean(all_heatmaps, axis=0, keepdims=True)
    mean_array = mean_array.reshape((40, 224, 224))
    return mean_array

def average_gradcam_heatmap_video(video_array, average_heatmap):
    nombre_video = r"C:\Users\ADMIN\Documents\xai_vd\try_videos"
    nombre_video = os.path.join(nombre_video, 'gradcam_heatmap_average' + '.avi')
    fps = 5  # Fotogramas por segundo del video
    alto, ancho = 224, 224
    # Configurar el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(nombre_video, fourcc, fps, (ancho, alto))
    
    print("###############")
    print("Average Heatmap")
    i=0
    for frame, heatmap in zip(video_array, average_heatmap):
        print(i)
        i = i+1

        # Grey image
        # gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
        # gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2RGB)

        # Get grad-cam heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_HOT)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        frame = frame.squeeze()
        imagen_superpuesta = cv2.addWeighted(frame, 0.2, heatmap, 0.8, 0)
        video.write(imagen_superpuesta)

    # Liberar recursos
    video.release()
    cv2.destroyAllWindows()

# Opacity
def rgba2rgb(rgba, background=(255,255,255)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    a = np.asarray( a, dtype='float32' ) / 255.0
    R, G, B = background
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B
    return np.asarray(rgb, dtype='uint8')

def loop_gradcam_opacity(vgg_model, layer_name, video_array):
    all_heatmaps = []
    for conv_layer_i in layer_name:
        print("##############")
        print(conv_layer_i)
        
        nombre_video = r"C:\Users\ADMIN\Documents\xai_vd\try_videos"
        nombre_video = os.path.join(nombre_video, 'gradcam_opacity_' + conv_layer_i + '.avi')
        fps = 5  # Fotogramas por segundo del video
        alto, ancho = 224, 224
        # Configurar el objeto VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(nombre_video, fourcc, fps, (ancho, alto))
        i = 0
        
        heatmaps_iter = []
        for frame in video_array:
            print(i)
            i = i+1
            # Get grad-cam heatmap
            heatmap = make_gradcam_heatmap(frame, vgg_model, conv_layer_i, pred_index=1)
            heatmap= (heatmap*255).astype(np.uint8)
            heatmaps_iter.append(heatmap)
            # Convert (None, 224, 224, 3) to (224, 224, 3)
            frame = frame.squeeze()
            # Convert to RGBA
            rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
            heatmap = cv2.resize(heatmap, (224, 224))
            rgba[:, :, 3] = heatmap
            final_rgba = rgba2rgb(rgba, background=(255,255,255))
            all_heatmaps.append(heatmap)
            video.write(final_rgba)
        # Liberar recursos
        video.release()
        cv2.destroyAllWindows()
    
    all_heatmaps = np.array(all_heatmaps)
    new_shape = (5, 40, 224, 224)
    all_heatmaps = all_heatmaps.reshape(new_shape)
    mean_array = np.mean(all_heatmaps, axis=0, keepdims=True)
    mean_array = mean_array.reshape((40, 224, 224))
    return mean_array

def average_gradcam_opacity_video(video_array, average_heatmap):
    nombre_video = r"C:\Users\ADMIN\Documents\xai_vd\try_videos"
    nombre_video = os.path.join(nombre_video, 'gradcam_opacity_average' + '.avi')
    fps = 5  # Fotogramas por segundo del video
    alto, ancho = 224, 224
    # Configurar el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(nombre_video, fourcc, fps, (ancho, alto))
    
    print("###############")
    print("Average Heatmap")
    i=0
    for frame, heatmap in zip(video_array, average_heatmap):
        print(i)
        i = i+1
        frame = frame.squeeze()
        frame = frame[..., ::-1]
        rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = heatmap
        final_rgba = rgba2rgb(rgba, background=(255,255,255))
        video.write(final_rgba)

    # Liberar recursos
    video.release()
    cv2.destroyAllWindows()

# Unify gradcam videos
def unify_gradcam_videos(videos, titulos):
    # Cargar los clips de video
    clips = [VideoFileClip(video) for video in videos]
    print(clips)
    # Función para agregar texto a un clip
    def add_title(clip, title):
        txt_clip = (TextClip(title, fontsize=14, color='green', font='Arial-Bold')
                    .set_position(('center', 0.9), relative=True)
                    .set_duration(clip.duration))
        return CompositeVideoClip([clip, txt_clip])

    # Añadir títulos a los clips
    clips_with_titles = [add_title(clip, titulo) for clip, titulo in zip(clips, titulos)]

    # Crear una cuadrícula con los clips de video
    final_clip = clips_array([
        [clips_with_titles[0], clips_with_titles[1], clips_with_titles[2]],
        [clips_with_titles[3], clips_with_titles[4], clips_with_titles[5]]
    ])

    # Guardar el video resultante
    final_clip.write_videofile(
        r"C:\Users\ADMIN\Desktop\GradCAM\total_gradcam.avi",
        codec='libx264', fps=24
    )

    # Cerrar los clips cargados
    for clip in clips:
        clip.close()

# MAIN
# # Grad-CAM for all layers and it's average
vgg_model = load_model(r"C:\Users\ADMIN\Mi unidad (pablo.negre@usal.es)\Papers\Violence detection\results\RWF-2000\VGG-19\train\biclass_vgg19.keras")
vgg_model = load_model(r"C:\Users\ADMIN\Mi unidad (pablo.negre@usal.es)\Papers\Violence detection\results\violent_flow\no-finetunning\VGG19\train VGG19\biclass_vgg19.keras")

layer_name = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']

titulos = ['Block 1 Conv 2', 
           'Block 2 Conv 2',
           'Block 3 Conv 4',
           'Block 4 Conv 4',
           'Block 5 Conv 4',
           'Average']

##### HEATMAP
def heatmap(vgg_model, video_path, target_frames):
    video_array = preprocess_single_video(video_path, target_frames)

    mean_array = loop_gradcam_heatmap(vgg_model, layer_name, video_array)
    average_gradcam_heatmap_video(video_array, mean_array)

    # Unificar Grad-CAM videos
    heatmap_videos = [r'C:\Users\ADMIN\Documents\xai_vd\try_videos\gradcam_heatmap_block1_conv2.avi', 
        r'C:\Users\ADMIN\Documents\xai_vd\try_videos\gradcam_heatmap_block2_conv2.avi',
        r'C:\Users\ADMIN\Documents\xai_vd\try_videos\gradcam_heatmap_block3_conv4.avi',
        r'C:\Users\ADMIN\Documents\xai_vd\try_videos\gradcam_heatmap_block4_conv4.avi',
        r'C:\Users\ADMIN\Documents\xai_vd\try_videos\gradcam_heatmap_block5_conv4.avi',
        r"C:\Users\ADMIN\Documents\xai_vd\try_videos\gradcam_heatmap_average.avi"]

    unify_gradcam_videos(heatmap_videos, titulos)

    # Eliminar videos de gradcam individuales
    for video in heatmap_videos:
        os.remove(video) 

##### OPACITY
def opacity(vgg_model, video_path, target_frames):
    video_array = preprocess_single_video(video_path, target_frames)

    mean_array = loop_gradcam_opacity(vgg_model, layer_name, video_array)
    mean_array = mean_array.reshape((40, 224, 224))
    average_gradcam_opacity_video(video_array, mean_array)

    opacity_videos = [r'C:\Users\ADMIN\Documents\xai_vd\try_videos\gradcam_opacity_block1_conv2.avi', 
                        r'C:\Users\ADMIN\Documents\xai_vd\try_videos\gradcam_opacity_block2_conv2.avi',
                        r'C:\Users\ADMIN\Documents\xai_vd\try_videos\gradcam_opacity_block3_conv4.avi',
                        r'C:\Users\ADMIN\Documents\xai_vd\try_videos\gradcam_opacity_block4_conv4.avi',
                        r'C:\Users\ADMIN\Documents\xai_vd\try_videos\gradcam_opacity_block5_conv4.avi',
                        r"C:\Users\ADMIN\Documents\xai_vd\try_videos\gradcam_opacity_average.avi"]
                        
    unify_gradcam_videos(opacity_videos, titulos)

    # Eliminar videos de gradcam individuales
    for video in opacity_videos:
        os.remove(video) 

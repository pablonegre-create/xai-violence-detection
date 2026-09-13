"""Grad-CAM over the frame-level backbone.

The maps are taken from the last convolutional layer of each MobileNetV2
block, not just the final one, because the blocks disagree in an informative
way: early blocks fire on contours, the last one on the interaction region.
"""

import numpy as np
import tensorflow as tf

# Last conv layer of each inverted-residual stage of keras MobileNetV2.
MOBILENETV2_BLOCK_LAYERS = [
    "block_2_add",
    "block_5_add",
    "block_9_add",
    "block_12_add",
    "out_relu",
]

VGG19_BLOCK_LAYERS = ["block%d_conv4" % i for i in range(2, 6)]


def make_gradcam(model, layer_name, img, class_index=None):
    """Heatmap for one image, normalised to [0, 1]."""
    grad_model = tf.keras.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output],
    )

    x = tf.convert_to_tensor(img[None] if img.ndim == 3 else img)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.nn.relu(cam).numpy()

    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def multi_block_gradcam(model, img, layers=None, class_index=None):
    """One map per block plus their mean, as used in the paper figure."""
    layers = layers or MOBILENETV2_BLOCK_LAYERS
    maps = {}
    for name in layers:
        try:
            maps[name] = make_gradcam(model, name, img, class_index)
        except ValueError:
            # layer missing for this backbone, skip rather than abort
            continue

    if maps:
        # resize everything to the largest map before averaging
        import cv2
        h = max(m.shape[0] for m in maps.values())
        w = max(m.shape[1] for m in maps.values())
        stack = [cv2.resize(m, (w, h)) for m in maps.values()]
        maps["mean"] = np.mean(stack, axis=0)
    return maps


def overlay(img, cam, alpha=0.4, colormap=None):
    """Heatmap on top of the frame, for the qualitative figures."""
    import cv2

    if colormap is None:
        colormap = cv2.COLORMAP_JET
    cam_r = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam_u8 = np.uint8(255 * np.clip(cam_r, 0, 1))
    heat = cv2.applyColorMap(cam_u8, colormap)

    base = img if img.dtype == np.uint8 else np.uint8(255 * np.clip(img, 0, 1))
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)

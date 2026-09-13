"""Model definitions for the two-stage violence detector.

The detector is split in two because that is what makes Grad-CAM usable: a
TimeDistributed wrapper around MobileNetV2 hides the inner layers and Keras
refuses to build the two-output submodel Grad-CAM needs ("Graph disconnected").
So the CNN runs frame by frame, the per-frame descriptors are stacked, and a
separate recurrent head consumes the stack.
"""

from tensorflow import keras
from tensorflow.keras import layers

# Frame descriptor size is fixed by the backbone; keep them here so the
# temporal head can be built without instantiating the CNN.
FEATURE_DIMS = {
    "mobilenetv2": 1280,
    "mobilenetv3small": 1024,
    "efficientnetb0": 1280,
    "resnet50": 2048,
    "vgg19": 512,
}


def build_backbone(name="mobilenetv2", input_size=128, weights="imagenet",
                   pooling="avg"):
    """Frame-level feature extractor, ImageNet-pretrained by default."""
    shape = (input_size, input_size, 3)
    kw = dict(include_top=False, weights=weights, input_shape=shape,
              pooling=pooling)

    if name == "mobilenetv2":
        net = keras.applications.MobileNetV2(**kw)
    elif name == "mobilenetv3small":
        net = keras.applications.MobileNetV3Small(**kw)
    elif name == "efficientnetb0":
        net = keras.applications.EfficientNetB0(**kw)
    elif name == "resnet50":
        net = keras.applications.ResNet50(**kw)
    elif name == "vgg19":
        net = keras.applications.VGG19(**kw)
    else:
        raise ValueError("unknown backbone: %s" % name)

    return net


def build_classifier_head(backbone, n_classes=2, dropout=0.3):
    """Backbone + dense head, used for the fine-tuning stage."""
    x = backbone.output
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="frame_cls")(x)
    return keras.Model(backbone.input, out, name=backbone.name + "_cls")


def build_temporal_head(n_frames=40, feature_dim=1280, lstm_units=(64, 32),
                        dense_units=(64, 32), batch_norm=False, dropout=0.3,
                        n_classes=2, name="bilstm_head"):
    """Bi-LSTM head over the stacked per-frame descriptors.

    lstm_units/dense_units are tuples so the ablation script can drop layers
    by passing shorter tuples.
    """
    inp = keras.Input(shape=(n_frames, feature_dim), name="features")
    x = inp

    for i, units in enumerate(lstm_units):
        last = i == len(lstm_units) - 1
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=not last),
            name="bilstm_%d" % (i + 1),
        )(x)
        if batch_norm:
            x = layers.BatchNormalization(name="bn_lstm_%d" % (i + 1))(x)

    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu", name="fc_%d" % (i + 1))(x)
        if batch_norm:
            x = layers.BatchNormalization(name="bn_fc_%d" % (i + 1))(x)
        if dropout:
            x = layers.Dropout(dropout, name="drop_%d" % (i + 1))(x)

    out = layers.Dense(n_classes, activation="softmax", name="video_cls")(x)
    return keras.Model(inp, out, name=name)


def build_gru_head(n_frames=40, feature_dim=1280, units=(64, 32), **kw):
    """GRU variant, only used in the recurrent-cell ablation."""
    inp = keras.Input(shape=(n_frames, feature_dim))
    x = inp
    for i, u in enumerate(units):
        last = i == len(units) - 1
        x = layers.Bidirectional(layers.GRU(u, return_sequences=not last))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(kw.get("n_classes", 2), activation="softmax")(x)
    return keras.Model(inp, out, name="bigru_head")


def build_pooling_head(n_frames=40, feature_dim=1280, n_classes=2):
    """No recurrence at all - average the frame descriptors.

    This is the "does the Bi-LSTM actually do anything" control.
    """
    inp = keras.Input(shape=(n_frames, feature_dim))
    x = layers.GlobalAveragePooling1D()(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="avgpool_head")


def set_finetune_depth(backbone, n_layers):
    """Freeze everything except the last `n_layers` layers.

    n_layers=0 -> feature extractor only; n_layers='all' -> train everything.
    """
    if n_layers == "all":
        for l in backbone.layers:
            l.trainable = True
        return backbone

    for l in backbone.layers:
        l.trainable = False
    if n_layers > 0:
        for l in backbone.layers[-n_layers:]:
            l.trainable = True
    return backbone

"""Parameter counts, FLOPs and wall-clock latency.

FLOPs come from the TF profiler on a concrete function traced at batch size 1.
The profiler reports multiply-accumulate pairs as two ops, which is the usual
convention in the Keras ecosystem; papers that quote MACs report roughly half
these numbers, so we print both to avoid ambiguity.

For a CNN+RNN video pipeline the per-clip cost is
    N_frames * cost(CNN on one frame) + cost(recurrent head on the sequence)
which is what total_clip_flops() returns.
"""

import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


def count_params(model):
    trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    non_trainable = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
        "fp32_mb": (trainable + non_trainable) * 4 / (1024 ** 2),
    }


def count_flops(model, input_shape=None):
    """FLOPs for one forward pass at batch size 1."""
    if input_shape is None:
        input_shape = model.input_shape[1:]

    @tf.function
    def fwd(x):
        return model(x, training=False)

    concrete = fwd.get_concrete_function(
        tf.TensorSpec([1] + list(input_shape), tf.float32))
    frozen = convert_variables_to_constants_v2(concrete)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts["output"] = "none"
    prof = tf.compat.v1.profiler.profile(
        graph=frozen.graph, run_meta=run_meta, cmd="scope", options=opts)
    return int(prof.total_float_ops)


def total_clip_flops(cnn_flops, head_flops, n_frames):
    return n_frames * cnn_flops + head_flops


def time_model(model, input_shape=None, n_warmup=10, n_runs=50, batch=1):
    """Median / mean / p95 latency in ms over n_runs forward passes."""
    if input_shape is None:
        input_shape = model.input_shape[1:]
    x = np.random.rand(batch, *input_shape).astype("float32")

    predict = tf.function(lambda t: model(t, training=False))
    xt = tf.constant(x)
    for _ in range(n_warmup):
        predict(xt)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict(xt)
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.asarray(times)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "n_runs": n_runs,
        "batch": batch,
    }


def peak_memory_mb():
    """Peak GPU memory if a GPU is visible, else None (CPU RSS is noisy)."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return None
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return info["peak"] / (1024 ** 2)
    except Exception:
        return None

"""Force the CUDA JIT cache to be populated.

The TensorFlow 2.15 wheel ships no sm_90 cubins, so on an H100 every kernel is
compiled from PTX the first time it runs. That costs tens of minutes, and it
happens again in every new process unless the result is cached on disk.

Run this once inside an interactive GPU session, with CUDA_CACHE_PATH pointing
somewhere in $HOME. It exercises the three kernel families the pipeline uses -
convolution forward, cuDNN LSTM forward, and the training backward pass - so
the real jobs start warm.

    export CUDA_CACHE_PATH=$HOME/.nv/ComputeCache
    export CUDA_CACHE_MAXSIZE=4294967296
    python scripts/warm_cuda_cache.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs visible:", gpus)
    if not gpus:
        print("No GPU: nothing to warm. Run this inside an srun session.")
        return 1

    cache = os.environ.get("CUDA_CACHE_PATH", "(default ~/.nv/ComputeCache)")
    print("CUDA_CACHE_PATH =", cache)
    print("CUDA_CACHE_MAXSIZE =", os.environ.get("CUDA_CACHE_MAXSIZE", "(default)"))
    print()

    from src.models.build import build_backbone, build_temporal_head

    t0 = time.perf_counter()
    print("1/3 convolution forward ...", flush=True)
    cnn = build_backbone("mobilenetv2", input_size=128, weights=None)
    cnn.predict(np.random.rand(8, 128, 128, 3).astype("float32"), verbose=0)
    print("    %.1f s" % (time.perf_counter() - t0), flush=True)

    t1 = time.perf_counter()
    print("2/3 recurrent forward ...", flush=True)
    head = build_temporal_head(40, 1280)
    head.predict(np.random.rand(4, 40, 1280).astype("float32"), verbose=0)
    print("    %.1f s" % (time.perf_counter() - t1), flush=True)

    t2 = time.perf_counter()
    print("3/3 training backward ...", flush=True)
    head.compile(optimizer=keras.optimizers.Adam(1e-3),
                 loss="categorical_crossentropy")
    x = np.random.rand(16, 40, 1280).astype("float32")
    y = keras.utils.to_categorical(np.random.randint(0, 2, 16), 2)
    head.fit(x, y, epochs=2, batch_size=8, verbose=0)
    print("    %.1f s" % (time.perf_counter() - t2), flush=True)

    print("\ntotal %.1f s" % (time.perf_counter() - t0))
    path = os.environ.get("CUDA_CACHE_PATH")
    if path and os.path.isdir(path):
        n = sum(len(f) for _, _, f in os.walk(path))
        size = sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(path) for f in fs)
        print("cache now holds %d files, %.1f MB" % (n, size / 1e6))
    print("Re-run this script: the second run should be much faster.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Corruptions applied to whole clips, for the robustness study.

Severity runs 1-5 following the ImageNet-C convention (Hendrycks &
Dietterich, ICLR 2019); the parameters below are the video analogues used in
spatio-temporal robustness benchmarks. Corruptions that have a temporal
character (camera shake, frame drop) are applied to the clip rather than
independently per frame, which is the point of using video.

Clips are float arrays (n_frames, h, w, 3) in [0, 1].
"""

import numpy as np

try:
    import cv2
except ImportError:  # keeps the module importable for docs/tests
    cv2 = None


def _clip(x):
    return np.clip(x, 0.0, 1.0).astype("float32")


def gaussian_noise(clip, severity=3):
    sigma = [0.04, 0.06, 0.10, 0.16, 0.26][severity - 1]
    rng = np.random.default_rng(0)
    return _clip(clip + rng.normal(0, sigma, clip.shape))


def low_illumination(clip, severity=3):
    """Gamma darkening plus the sensor noise that comes with low light."""
    gamma = [1.6, 2.2, 3.0, 4.0, 5.5][severity - 1]
    gain = [0.7, 0.55, 0.42, 0.32, 0.22][severity - 1]
    rng = np.random.default_rng(1)
    out = np.power(clip, gamma) * gain
    return _clip(out + rng.normal(0, 0.02 * severity, clip.shape))


def motion_blur(clip, severity=3):
    k = [3, 5, 7, 11, 15][severity - 1]
    kernel = np.zeros((k, k), dtype="float32")
    kernel[k // 2, :] = 1.0 / k
    return _clip(np.stack([cv2.filter2D(f, -1, kernel) for f in clip]))


def camera_shake(clip, severity=3):
    """Random per-frame translation/rotation, smooth in time."""
    amp = [1.5, 3.0, 5.0, 8.0, 12.0][severity - 1]
    rng = np.random.default_rng(2)
    n, h, w = clip.shape[:3]

    # low-pass a random walk so the jitter looks like handheld motion
    dx = np.cumsum(rng.normal(0, amp / 3, n))
    dy = np.cumsum(rng.normal(0, amp / 3, n))
    ang = np.cumsum(rng.normal(0, amp / 12, n))
    dx -= dx.mean(); dy -= dy.mean(); ang -= ang.mean()

    out = []
    for i, f in enumerate(clip):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(ang[i]), 1.0)
        M[0, 2] += dx[i]
        M[1, 2] += dy[i]
        out.append(cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    return _clip(np.stack(out))


def occlusion(clip, severity=3, static=True):
    """Rectangular cut-out. Static mimics a fixed obstruction (a pillar, a
    sign); moving mimics someone walking through the foreground."""
    frac = [0.10, 0.20, 0.30, 0.42, 0.55][severity - 1]
    rng = np.random.default_rng(3)
    n, h, w = clip.shape[:3]
    bh, bw = int(h * np.sqrt(frac)), int(w * np.sqrt(frac))

    out = clip.copy()
    y0, x0 = rng.integers(0, max(1, h - bh)), rng.integers(0, max(1, w - bw))
    for i in range(n):
        if static:
            y, x = y0, x0
        else:
            y = int(np.clip(y0 + 0.5 * i, 0, h - bh))
            x = int(np.clip(x0 + 1.5 * i, 0, w - bw))
        out[i, y:y + bh, x:x + bw] = 0.0
    return _clip(out)


def compression(clip, severity=3):
    """JPEG per frame, as a cheap stand-in for aggressive H.264 recompression
    on a surveillance link."""
    q = [70, 50, 35, 20, 10][severity - 1]
    out = []
    for f in clip:
        u8 = np.uint8(255 * f)
        ok, enc = cv2.imencode(".jpg", u8, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        out.append(cv2.imdecode(enc, cv2.IMREAD_COLOR).astype("float32") / 255.)
    return _clip(np.stack(out))


def frame_drop(clip, severity=3):
    """Drop frames and hold the previous one, as a lossy stream would."""
    p = [0.05, 0.10, 0.20, 0.35, 0.50][severity - 1]
    rng = np.random.default_rng(4)
    out = clip.copy()
    for i in range(1, len(clip)):
        if rng.random() < p:
            out[i] = out[i - 1]
    return out


def downscale(clip, severity=3):
    """Resolution loss from a distant or cheap camera."""
    f = [0.75, 0.5, 0.35, 0.25, 0.15][severity - 1]
    n, h, w = clip.shape[:3]
    small = [cv2.resize(x, (max(2, int(w * f)), max(2, int(h * f))))
             for x in clip]
    return _clip(np.stack([cv2.resize(x, (w, h)) for x in small]))


def rain_streaks(clip, severity=3):
    """Crude weather simulation: bright oriented streaks plus a haze veil."""
    n_drops = [200, 500, 900, 1400, 2000][severity - 1]
    rng = np.random.default_rng(5)
    n, h, w = clip.shape[:3]
    out = clip.copy()
    for i in range(n):
        layer = np.zeros((h, w), dtype="float32")
        xs = rng.integers(0, w, n_drops)
        ys = rng.integers(0, h, n_drops)
        length = rng.integers(4, 12, n_drops)
        for x, y, L in zip(xs, ys, length):
            cv2.line(layer, (int(x), int(y)), (int(x - 2), int(min(h - 1, y + L))),
                     0.8, 1)
        layer = cv2.GaussianBlur(layer, (3, 3), 0)
        out[i] = np.clip(out[i] * 0.92 + 0.08 + layer[..., None] * 0.5, 0, 1)
    return _clip(out)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "low_illumination": low_illumination,
    "motion_blur": motion_blur,
    "camera_shake": camera_shake,
    "occlusion_static": lambda c, s: occlusion(c, s, static=True),
    "occlusion_moving": lambda c, s: occlusion(c, s, static=False),
    "compression": compression,
    "frame_drop": frame_drop,
    "downscale": downscale,
    "rain": rain_streaks,
}


def mean_corruption_error(acc_corrupt, acc_clean):
    """Relative degradation, 0 = untouched, 1 = all accuracy above chance lost."""
    if acc_clean <= 0.5:
        return float("nan")
    return float((acc_clean - acc_corrupt) / (acc_clean - 0.5))


# ---------------------------------------------------------------- adversarial

def fgsm(clip, model, label, eps=2.0 / 255.0, class_index=1):
    """One-step FGSM in pixel space against the frame classifier."""
    import tensorflow as tf

    x = tf.convert_to_tensor(clip[None] if clip.ndim == 3 else clip)
    y = tf.one_hot([label] * x.shape[0], model.output_shape[-1])
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x, training=False)
        loss = tf.keras.losses.categorical_crossentropy(y, pred)
    g = tape.gradient(loss, x)
    return _clip((x + eps * tf.sign(g)).numpy())


def pgd(clip, model, label, eps=2.0 / 255.0, alpha=0.5 / 255.0, steps=10):
    import tensorflow as tf

    x0 = tf.convert_to_tensor(clip[None] if clip.ndim == 3 else clip)
    x = tf.identity(x0)
    y = tf.one_hot([label] * x.shape[0], model.output_shape[-1])
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = tf.keras.losses.categorical_crossentropy(y, model(x, training=False))
        g = tape.gradient(loss, x)
        x = x + alpha * tf.sign(g)
        x = tf.clip_by_value(x, x0 - eps, x0 + eps)
        x = tf.clip_by_value(x, 0.0, 1.0)
    return x.numpy()

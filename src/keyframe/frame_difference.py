"""Frame differencing for keyframe selection.

The mean absolute difference between consecutive greyscale frames is a cheap
proxy for how much is happening. It is not a violence detector - a camera pan
or a light switch will spike it - which is exactly why it is paired with the
person count rather than used alone.

The threshold is scene-dependent: a fight filling the frame gives a mean
difference above ~40 grey levels, the same fight at 20 m gives single digits.
estimate_threshold() derives it from a quiet stretch of the same camera
instead of hard-coding a number.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def frame_difference(prev_gray, cur_gray):
    """Absolute difference map plus its summary statistics."""
    diff = cv2.absdiff(cur_gray, prev_gray)
    return diff, {
        "mean": float(diff.mean()),
        "median": float(np.median(diff)),
        "std": float(diff.std()),
        "min": int(diff.min()),
        "max": int(diff.max()),
        "p95": float(np.percentile(diff, 95)),
    }


def motion_profile(video_path, stride=1, resize=None):
    """Per-frame motion score for a whole video."""
    cap = cv2.VideoCapture(video_path)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("cannot read %s" % video_path)

    if resize:
        prev = cv2.resize(prev, resize)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    scores, idx, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % stride:
            continue
        if resize:
            frame = cv2.resize(frame, resize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, st = frame_difference(prev_gray, gray)
        scores.append(st["mean"])
        idx.append(i)
        prev_gray = gray

    cap.release()
    return np.asarray(idx), np.asarray(scores)


def estimate_threshold(scores, k=3.0, quantile=0.5):
    """Robust threshold from the quiet half of the sequence.

    median + k * MAD over the frames below `quantile`, so a long fight in the
    middle of the clip does not inflate its own threshold.
    """
    s = np.asarray(scores, float)
    quiet = s[s <= np.quantile(s, quantile)]
    if quiet.size == 0:
        quiet = s
    med = np.median(quiet)
    mad = np.median(np.abs(quiet - med)) * 1.4826
    return float(med + k * max(mad, 1e-6))


def select_keyframes(scores, threshold=None, min_gap=5):
    """Indices whose motion exceeds the threshold, thinned by min_gap."""
    s = np.asarray(scores, float)
    if threshold is None:
        threshold = estimate_threshold(s)

    picked, last = [], -10 ** 9
    for i, v in enumerate(s):
        if v >= threshold and i - last >= min_gap:
            picked.append(i)
            last = i
    return np.asarray(picked), threshold

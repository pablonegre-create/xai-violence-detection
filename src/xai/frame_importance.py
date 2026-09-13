"""Temporal frame-importance by block ablation.

The detector takes a fixed-length sequence of frame descriptors. To find out
which part of the clip carries the decision we drop a contiguous block of
frames, pad the sequence back to its original length by repeating the last
frame, and look at how far the violence score falls.

Because the CNN descriptors are computed once and cached, each ablated variant
costs only a forward pass of the recurrent head, which is why the whole sweep
runs in well under a second per clip.
"""

import numpy as np

EPS = 1e-8


def _pad_to_length(seq, n):
    """Repeat the last frame until the sequence is n long."""
    if len(seq) >= n:
        return seq[:n]
    pad = np.repeat(seq[-1][None], n - len(seq), axis=0)
    return np.concatenate([seq, pad], axis=0)


def ablate_block(features, start, block):
    """Drop features[start:start+block] and pad back to the original length."""
    n = len(features)
    kept = np.concatenate([features[:start], features[start + block:]], axis=0)
    return _pad_to_length(kept, n)


def frame_importance(features, head, block=5, stride=None, class_index=1,
                     batch_size=64):
    """Importance curve for one clip.

    features : (n_frames, d) cached CNN descriptors
    head     : Keras model mapping (1, n_frames, d) -> class probabilities
    block    : number of consecutive frames removed per step
    stride   : step between block starts (defaults to 1, i.e. sliding)

    Returns dict with the per-position importance I_i = p_full / p_ablated
    and the raw ablated scores. I_i > 1 means removing that block hurt the
    prediction, so the block mattered.
    """
    features = np.asarray(features, dtype="float32")
    n = len(features)
    stride = 1 if stride is None else stride

    p_full = float(head.predict(features[None], verbose=0)[0][class_index])

    starts = list(range(0, max(1, n - block + 1), stride))
    variants = np.stack([ablate_block(features, s, block) for s in starts])

    probs = head.predict(variants, batch_size=batch_size,
                         verbose=0)[:, class_index]
    importance = p_full / np.maximum(probs, EPS)

    # spread the block score over the frames it covers so the curve is
    # per-frame and comparable with attribution methods
    per_frame = np.zeros(n)
    hits = np.zeros(n)
    for s, imp in zip(starts, importance):
        per_frame[s:s + block] += imp
        hits[s:s + block] += 1
    per_frame = np.divide(per_frame, np.maximum(hits, 1))

    return {
        "p_full": p_full,
        "block_starts": np.asarray(starts),
        "block_importance": importance,
        "ablated_prob": probs,
        "per_frame_importance": per_frame,
        "n_model_calls": len(starts) + 1,
    }


def importance_attribution(features, head, block=5, class_index=1):
    """Signed attribution in the same units as other XAI methods.

    I_i is a ratio, which is convenient to read but awkward to compare with
    additive attributions. This returns the plain probability drop
    p_full - p_ablated, which is on the same scale as occlusion and
    Shapley-style attributions and is what the quantitative comparison uses.
    """
    r = frame_importance(features, head, block=block, class_index=class_index)
    drop = r["p_full"] - r["ablated_prob"]

    n = len(features)
    per_frame = np.zeros(n)
    hits = np.zeros(n)
    for s, d in zip(r["block_starts"], drop):
        per_frame[s:s + block] += d
        hits[s:s + block] += 1
    return np.divide(per_frame, np.maximum(hits, 1))


def sweep_block_sizes(features_list, head, blocks=(1, 2, 5, 10),
                      class_index=1):
    """Reproduces the block-size study: how stable is the curve as the block
    grows, and what does it cost."""
    import time

    out = []
    for b in blocks:
        means, stds, times = [], [], []
        for feats in features_list:
            t0 = time.perf_counter()
            r = frame_importance(feats, head, block=b, class_index=class_index)
            times.append(time.perf_counter() - t0)
            means.append(float(np.mean(r["block_importance"])))
            stds.append(float(np.std(r["block_importance"])))
        out.append({
            "block": b,
            "median_mean_importance": float(np.median(means)),
            "sd_across_videos": float(np.std(means)),
            "mean_sd_within_video": float(np.mean(stds)),
            "median_sd_within_video": float(np.median(stds)),
            "mean_time_s": float(np.mean(times)),
        })
    return out

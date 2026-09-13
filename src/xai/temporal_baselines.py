"""Temporal attribution baselines the proposed block-ablation is compared to.

All of them return one score per frame for a single clip, computed on the
cached CNN descriptors so the comparison is like for like: every method sees
the same (n_frames, d) input and the same recurrent head.

Sign convention: positive = frame supports the predicted class.
"""

import numpy as np

EPS = 1e-8


def _predict(head, seqs, class_index, batch_size=64):
    seqs = np.asarray(seqs, dtype="float32")
    if seqs.ndim == 2:
        seqs = seqs[None]
    return head.predict(seqs, batch_size=batch_size, verbose=0)[:, class_index]


def temporal_occlusion(features, head, class_index=1, baseline="zero",
                       window=1):
    """Classic occlusion: replace a window of frames with a baseline value.

    Differs from the proposed method in that the sequence length never
    changes - frames are masked in place rather than removed and re-padded.
    """
    features = np.asarray(features, dtype="float32")
    n = len(features)
    p_full = _predict(head, features, class_index)[0]

    if baseline == "zero":
        fill = np.zeros(features.shape[1], dtype="float32")
    elif baseline == "mean":
        fill = features.mean(axis=0)
    else:
        raise ValueError(baseline)

    variants = []
    for i in range(n):
        v = features.copy()
        lo, hi = max(0, i - window // 2), min(n, i + window // 2 + 1)
        v[lo:hi] = fill
        variants.append(v)

    probs = _predict(head, np.stack(variants), class_index)
    return p_full - probs


def temporal_integrated_gradients(features, head, class_index=1, steps=32,
                                  baseline=None):
    """Integrated Gradients along the time axis, summed over feature dims."""
    import tensorflow as tf

    features = np.asarray(features, dtype="float32")
    if baseline is None:
        baseline = np.zeros_like(features)

    alphas = np.linspace(0.0, 1.0, steps, dtype="float32")
    path = np.stack([baseline + a * (features - baseline) for a in alphas])
    path_t = tf.convert_to_tensor(path)

    with tf.GradientTape() as tape:
        tape.watch(path_t)
        preds = head(path_t, training=False)[:, class_index]
    grads = tape.gradient(preds, path_t).numpy()

    avg_grad = grads.mean(axis=0)
    ig = (features - baseline) * avg_grad
    return ig.sum(axis=1)


def temporal_gradient_input(features, head, class_index=1):
    """Plain gradient x input, cheapest possible attribution."""
    import tensorflow as tf

    features = np.asarray(features, dtype="float32")[None]
    x = tf.convert_to_tensor(features)
    with tf.GradientTape() as tape:
        tape.watch(x)
        p = head(x, training=False)[:, class_index]
    g = tape.gradient(p, x).numpy()[0]
    return (features[0] * g).sum(axis=1)


def temporal_shapley_sampling(features, head, class_index=1, n_samples=200,
                              rng=None, group=5):
    """Monte-Carlo Shapley values over frame groups.

    This is the honest stand-in for TimeSHAP when the reference implementation
    cannot be used: TimeSHAP computes event-level Shapley values by KernelSHAP
    over binary coalitions of events, with pruning. Here events are contiguous
    groups of `group` frames, coalitions are sampled uniformly, and an absent
    group is replaced by the sequence mean (TimeSHAP's background strategy).

    Cost is n_samples model calls, versus n_frames/stride for block ablation -
    the point of the comparison.
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    features = np.asarray(features, dtype="float32")
    n = len(features)
    groups = [(i, min(n, i + group)) for i in range(0, n, group)]
    m = len(groups)
    background = features.mean(axis=0)

    phi = np.zeros(m)
    counts = np.zeros(m)

    for _ in range(n_samples):
        order = rng.permutation(m)
        present = np.zeros(m, dtype=bool)

        def build(mask):
            v = features.copy()
            for gi, (lo, hi) in enumerate(groups):
                if not mask[gi]:
                    v[lo:hi] = background
            return v

        prev = _predict(head, build(present), class_index)[0]
        for gi in order:
            present[gi] = True
            cur = _predict(head, build(present), class_index)[0]
            phi[gi] += cur - prev
            counts[gi] += 1
            prev = cur

    phi = phi / np.maximum(counts, 1)

    per_frame = np.zeros(n)
    for gi, (lo, hi) in enumerate(groups):
        per_frame[lo:hi] = phi[gi] / (hi - lo)
    return per_frame


def timeshap_event_attribution(features, head, class_index=1, nsamples=320,
                               pruning_tol=0.025):
    """Wrapper around the reference TimeSHAP package if it is installed.

    Falls back to temporal_shapley_sampling with a warning otherwise, so the
    comparison script still runs on machines without the dependency.
    """
    try:
        from timeshap.explainer import local_event
    except ImportError:
        import warnings
        warnings.warn("timeshap not installed, using sampling-based Shapley")
        return temporal_shapley_sampling(features, head,
                                         class_index=class_index,
                                         n_samples=max(1, nsamples // 8))

    features = np.asarray(features, dtype="float32")

    def f(x):
        # timeshap expects (batch, seq, feat) -> (batch, 1)
        return head.predict(x, verbose=0)[:, class_index:class_index + 1]

    baseline = features.mean(axis=0, keepdims=True)[None]
    out = local_event(f, features[None],
                      {"rs": 0, "nsamples": nsamples},
                      None, baseline,
                      {"tol": pruning_tol})

    n = len(features)
    per_frame = np.zeros(n)
    # local_event returns a dataframe with one row per event index counted
    # backwards from the end of the sequence
    for _, row in out.iterrows():
        try:
            idx = int(row["Feature"].replace("Event ", ""))
        except (ValueError, AttributeError):
            continue
        pos = n + idx if idx < 0 else n - idx
        if 0 <= pos < n:
            per_frame[pos] = float(row["Shapley Value"])
    return per_frame


METHODS = {
    "block_ablation": None,  # filled in by the eval script
    "occlusion": temporal_occlusion,
    "integrated_gradients": temporal_integrated_gradients,
    "gradient_input": temporal_gradient_input,
    "shapley_sampling": temporal_shapley_sampling,
    "timeshap": timeshap_event_attribution,
}

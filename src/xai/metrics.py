"""Quantitative evaluation of explanations.

Two families are implemented:

  * temporal - deletion/insertion curves over frames, plus rank agreement
    between attribution methods. Used to compare block ablation against
    TimeSHAP and the other temporal baselines.

  * spatial - deletion/insertion over image pixels, average drop / increase
    in confidence, and a pointing-game style hit rate against person boxes.
    Used to put a number on the Grad-CAM maps.

Deletion/insertion follow Petsiuk et al. (RISE, BMVC 2018): order the elements
by attribution, remove (or add) them a step at a time, and integrate the class
probability over the fraction removed. Lower deletion AUC and higher insertion
AUC mean the attribution ranked the evidence better.
"""

import numpy as np

EPS = 1e-8


# ---------------------------------------------------------------- temporal

def temporal_deletion_curve(features, head, attribution, class_index=1,
                            baseline="mean", order="descending"):
    """Blank frames one at a time in attribution order, track the score."""
    features = np.asarray(features, dtype="float32")
    n = len(features)
    fill = features.mean(axis=0) if baseline == "mean" else np.zeros(
        features.shape[1], dtype="float32")

    idx = np.argsort(attribution)
    if order == "descending":
        idx = idx[::-1]

    seqs = [features.copy()]
    cur = features.copy()
    for i in idx:
        cur = cur.copy()
        cur[i] = fill
        seqs.append(cur)

    probs = head.predict(np.stack(seqs), batch_size=64,
                         verbose=0)[:, class_index]
    return probs


def temporal_insertion_curve(features, head, attribution, class_index=1,
                             baseline="mean"):
    """Start from a blank clip and restore frames in attribution order."""
    features = np.asarray(features, dtype="float32")
    n = len(features)
    fill = features.mean(axis=0) if baseline == "mean" else np.zeros(
        features.shape[1], dtype="float32")

    idx = np.argsort(attribution)[::-1]
    blank = np.repeat(fill[None], n, axis=0)

    seqs = [blank.copy()]
    cur = blank.copy()
    for i in idx:
        cur = cur.copy()
        cur[i] = features[i]
        seqs.append(cur)

    probs = head.predict(np.stack(seqs), batch_size=64,
                         verbose=0)[:, class_index]
    return probs


def auc(curve):
    """Normalised area under a curve sampled at equal steps."""
    curve = np.asarray(curve, dtype="float64")
    return float(np.trapz(curve, dx=1.0 / max(1, len(curve) - 1)))


def rank_agreement(a, b, k=None):
    """Spearman rho and top-k overlap between two attribution vectors."""
    from scipy.stats import spearmanr, kendalltau

    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    rho = spearmanr(a, b).correlation
    tau = kendalltau(a, b).correlation
    out = {"spearman": float(rho), "kendall": float(tau)}

    if k:
        ta = set(np.argsort(a)[::-1][:k].tolist())
        tb = set(np.argsort(b)[::-1][:k].tolist())
        out["top%d_jaccard" % k] = len(ta & tb) / len(ta | tb)
    return out


# ----------------------------------------------------------------- spatial

def _mask_pixels(img, order, n_remove, fill):
    flat = img.reshape(-1, img.shape[-1]).copy()
    flat[order[:n_remove]] = fill
    return flat.reshape(img.shape)


def spatial_deletion_insertion(img, model, heatmap, class_index=1, steps=50,
                               fill_value=0.0):
    """Pixel-level deletion and insertion AUC for one frame."""
    img = np.asarray(img, dtype="float32")
    h, w = heatmap.shape[:2]
    if (h, w) != img.shape[:2]:
        import cv2
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    order = np.argsort(heatmap.reshape(-1))[::-1]
    total = order.size
    cuts = np.linspace(0, total, steps + 1).astype(int)
    fill = np.full(img.shape[-1], fill_value, dtype="float32")

    dele, ins = [], []
    blank = np.full_like(img, fill_value)
    for c in cuts:
        dele.append(_mask_pixels(img, order, c, fill))
        keep = blank.reshape(-1, img.shape[-1]).copy()
        keep[order[:c]] = img.reshape(-1, img.shape[-1])[order[:c]]
        ins.append(keep.reshape(img.shape))

    pd = model.predict(np.stack(dele), batch_size=16,
                       verbose=0)[:, class_index]
    pi = model.predict(np.stack(ins), batch_size=16,
                       verbose=0)[:, class_index]
    return {"deletion": pd, "insertion": pi,
            "deletion_auc": auc(pd), "insertion_auc": auc(pi)}


def average_drop_increase(img, model, heatmap, class_index=1, percentile=50):
    """Grad-CAM++ style Average Drop and Increase in Confidence.

    Keep only the most salient region (top `percentile` of the heatmap) and
    see whether the score falls (drop) or rises (increase).
    """
    import cv2

    img = np.asarray(img, dtype="float32")
    hm = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    thr = np.percentile(hm, percentile)
    masked = img * (hm >= thr)[..., None]

    p_full = float(model.predict(img[None], verbose=0)[0][class_index])
    p_mask = float(model.predict(masked[None], verbose=0)[0][class_index])

    drop = max(0.0, p_full - p_mask) / max(p_full, EPS)
    return {"p_full": p_full, "p_masked": p_mask,
            "average_drop": drop, "increase": int(p_mask > p_full)}


def pointing_game(heatmap, boxes, tolerance=0):
    """Does the heatmap peak fall inside any of the given boxes?

    boxes are (x1, y1, x2, y2) in heatmap coordinates. In this work the boxes
    come from the YOLO person detector, so a hit means the model is looking at
    a person rather than at background. That is a weak proxy for "looking at
    the violence" - it is a necessary condition, not a sufficient one - and is
    reported as such.
    """
    if not len(boxes):
        return None
    iy, ix = np.unravel_index(np.argmax(heatmap), heatmap.shape[:2])
    for x1, y1, x2, y2 in boxes:
        if (x1 - tolerance <= ix <= x2 + tolerance and
                y1 - tolerance <= iy <= y2 + tolerance):
            return 1
    return 0


def energy_pointing_game(heatmap, boxes):
    """Fraction of total heatmap energy that falls inside the boxes.

    Less brittle than the argmax version - a map that is broadly right but
    whose single peak is off scores poorly on pointing game and sensibly here.
    """
    if not len(boxes):
        return None
    hm = np.clip(np.asarray(heatmap, dtype="float64"), 0, None)
    total = hm.sum()
    if total <= EPS:
        return 0.0
    mask = np.zeros(hm.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[int(y1):int(y2), int(x1):int(x2)] = True
    return float(hm[mask].sum() / total)


# ------------------------------------------------------------ sanity check

def model_randomisation_test(img, model_fn, heatmap_fn, layers_to_randomise,
                             metric="spearman"):
    """Adebayo et al. (2018) cascading randomisation.

    Randomise weights layer by layer from the top down and measure how fast the
    explanation decorrelates from the original. A map that barely changes is
    acting as an edge detector, not as an explanation.
    """
    from scipy.stats import spearmanr

    base = heatmap_fn(model_fn(), img).reshape(-1)
    out = []
    model = model_fn()
    for name in layers_to_randomise:
        layer = model.get_layer(name)
        layer.set_weights([np.random.normal(0, 0.05, w.shape)
                           for w in layer.get_weights()])
        hm = heatmap_fn(model, img).reshape(-1)
        rho = spearmanr(base, hm).correlation
        out.append({"layer": name, metric: float(rho)})
    return out

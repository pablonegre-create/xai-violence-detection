"""Attribution on real footage with a temporal event we placed ourselves.

The synthetic benchmark plants an event in Gaussian noise. It establishes that
the attribution methods can find a localised temporal signal at all, which is
the control the paper needs, but as a reviewer pointed out the task is far
easier than real violence dynamics: the background has no structure to confuse
the explainer.

This builds the same experiment on real video. A clip is a real non-violent
RLVS clip into which a contiguous window of frames from a donor clip has been
spliced at a known offset. For positives the donor is violent; for negatives it
is a *different non-violent* clip. Both classes therefore contain a splice of
the same length at the same offsets, so the cut itself carries no label
information and the head has to use what is inside the window. The ground-truth
attribution mask is exactly that window.

Two phases, so the expensive one runs once:

    python scripts/run_semisynthetic_xai.py build --root $DATA_ROOT/rlvs \\
        --backbone-weights ckpt/mnv2_ft5.h5 --out features/semisynth_rlvs
    python scripts/run_semisynthetic_xai.py score --data features/semisynth_rlvs \\
        --out results/semisynthetic_xai.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# --------------------------------------------------------------- build phase

def build(args):
    from src.data.datasets import index_flat, load_clip
    from src.models.build import build_backbone

    items = index_flat(args.root)
    viol = [p for p, l in items if l == 1]
    peace = [p for p, l in items if l == 0]
    print("%d violent, %d non-violent source clips" % (len(viol), len(peace)))
    if len(peace) < 2 or not viol:
        raise SystemExit("need both classes to splice")

    cnn = build_backbone(args.backbone, input_size=args.input_size,
                         weights="imagenet")
    if args.backbone_weights:
        cnn.load_weights(args.backbone_weights, by_name=True,
                         skip_mismatch=True)
        print("loaded", args.backbone_weights)

    rng = np.random.default_rng(args.seed)
    feats, labels, masks, recipe = [], [], [], []
    n_done = 0
    attempts = 0
    cache = {}

    def clip_of(path):
        if path not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[path] = load_clip(path, n_frames=args.frames,
                                    size=args.input_size)
        return cache[path]

    while n_done < args.n and attempts < args.n * 4:
        attempts += 1
        label = n_done % 2                      # balanced by construction
        bg_path = peace[rng.integers(len(peace))]
        if label == 1:
            donor_path = viol[rng.integers(len(viol))]
        else:
            donor_path = peace[rng.integers(len(peace))]
            if donor_path == bg_path:
                continue
        try:
            bg = clip_of(bg_path).copy()
            donor = clip_of(donor_path)
        except Exception as e:
            print("skip (%s)" % e)
            continue

        off = int(rng.integers(0, args.frames - args.event_len + 1))
        bg[off:off + args.event_len] = donor[off:off + args.event_len]

        f = cnn.predict(bg, batch_size=args.frames, verbose=0)
        mask = np.zeros(args.frames, dtype="int8")
        mask[off:off + args.event_len] = 1

        feats.append(f.astype("float32"))
        labels.append(label)
        masks.append(mask)
        recipe.append({"background": os.path.relpath(bg_path, args.root),
                       "donor": os.path.relpath(donor_path, args.root),
                       "offset": off, "label": label})
        n_done += 1
        if n_done % 50 == 0:
            print("  %d/%d" % (n_done, args.n), flush=True)

    X = np.stack(feats)
    y = np.asarray(labels)
    M = np.stack(masks)

    # a plain split: the clips are constructed, so there is nothing to leak
    # except a background reused across both sides, which the head cannot use
    # because both classes draw backgrounds from the same pool
    n_tr = int(0.8 * len(y))
    splits = np.array(["train"] * n_tr + ["test"] * (len(y) - n_tr))

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "features.npy"), X)
    np.save(os.path.join(args.out, "labels.npy"), y)
    np.save(os.path.join(args.out, "masks.npy"), M)
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump({"splits": splits.tolist(), "event_len": args.event_len,
                   "frames": args.frames, "input_size": args.input_size,
                   "weights": args.backbone_weights, "source": args.root,
                   "recipe": recipe}, fh, indent=2)
    print("wrote %d spliced clips to %s  (%d positive)"
          % (len(y), args.out, int(y.sum())))


# --------------------------------------------------------------- score phase

def score_attribution(attr, gt_mask):
    from scipy.stats import spearmanr
    attr = np.asarray(attr, dtype="float64")
    k = int(gt_mask.sum())
    if k == 0:
        return None
    top = np.argsort(attr)[::-1][:k]
    rho = spearmanr(attr, gt_mask).correlation
    pos = np.clip(attr, 0, None)
    return {"precision_at_k": float(gt_mask[top].sum() / k),
            "pointing": float(gt_mask[int(np.argmax(attr))]),
            "spearman_gt": float(rho) if rho == rho else 0.0,
            "mass_on_event": float(pos[gt_mask > 0].sum()
                                   / max(pos.sum(), 1e-9))}


def score(args):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import accuracy_score
    from src.models.build import build_temporal_head
    from src.xai import temporal_baselines as tb
    from src.xai.frame_importance import importance_attribution
    from src.xai.metrics import (temporal_deletion_curve,
                                 temporal_insertion_curve, auc)

    X = np.load(os.path.join(args.data, "features.npy"))
    y = np.load(os.path.join(args.data, "labels.npy"))
    M = np.load(os.path.join(args.data, "masks.npy"))
    meta = json.load(open(os.path.join(args.data, "index.json")))
    split = np.asarray(meta["splits"])
    tr, te = split == "train", split == "test"
    n_frames, dim = X.shape[1], X.shape[2]
    print("%d clips, %d frames, %d dims; %d train / %d test"
          % (len(y), n_frames, dim, tr.sum(), te.sum()))

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    head = build_temporal_head(n_frames, dim)
    head.compile(optimizer=keras.optimizers.Adam(1e-3),
                 loss="categorical_crossentropy", metrics=["accuracy"])
    Y = keras.utils.to_categorical(y, 2)
    head.fit(X[tr], Y[tr], validation_split=0.15, epochs=args.epochs,
             batch_size=16, verbose=0,
             callbacks=[keras.callbacks.EarlyStopping(
                 monitor="val_loss", patience=10, restore_best_weights=True)])
    acc = float(accuracy_score(y[te], head.predict(X[te], verbose=0).argmax(1)))
    print("head test accuracy on the spliced task: %.4f" % acc)
    if acc < 0.6:
        print("WARNING: the head barely learns this task, so the attribution "
              "scores below describe a model that has little to attribute")

    pos = [i for i in np.where(te)[0] if y[i] == 1][:args.explain]
    print("explaining %d positive test clips" % len(pos))

    methods = {
        "block_ablation": lambda f: importance_attribution(f, head,
                                                           block=args.block),
        "occlusion": lambda f: tb.temporal_occlusion(f, head),
        "gradient_input": lambda f: tb.temporal_gradient_input(f, head),
        "integrated_gradients": lambda f: tb.temporal_integrated_gradients(
            f, head),
        "shapley_sampling": lambda f: tb.temporal_shapley_sampling(
            f, head, n_samples=args.shapley_samples, group=args.block),
    }

    rows = {m: {"scores": [], "time": [], "del": [], "ins": []}
            for m in list(methods) + ["random"]}
    rng = np.random.default_rng(args.seed)

    for n_done, i in enumerate(pos):
        feats, mask = X[i], M[i]
        for name, fn in methods.items():
            t0 = time.perf_counter()
            attr = fn(feats)
            dt = time.perf_counter() - t0
            s = score_attribution(attr, mask)
            if s:
                rows[name]["scores"].append(s)
                rows[name]["time"].append(dt)
                rows[name]["del"].append(
                    auc(temporal_deletion_curve(feats, head, attr)))
                rows[name]["ins"].append(
                    auc(temporal_insertion_curve(feats, head, attr)))
        attr = rng.normal(size=n_frames)
        s = score_attribution(attr, mask)
        rows["random"]["scores"].append(s)
        rows["random"]["time"].append(0.0)
        rows["random"]["del"].append(
            auc(temporal_deletion_curve(feats, head, attr)))
        rows["random"]["ins"].append(
            auc(temporal_insertion_curve(feats, head, attr)))
        if (n_done + 1) % 10 == 0:
            print("  %d/%d" % (n_done + 1, len(pos)), flush=True)

    agg = {}
    for name, d in rows.items():
        if not d["scores"]:
            continue
        keys = d["scores"][0].keys()
        a = {k + "_mean": float(np.mean([s[k] for s in d["scores"]]))
             for k in keys}
        a.update({k + "_std": float(np.std([s[k] for s in d["scores"]]))
                  for k in keys})
        a["time_s_mean"] = float(np.mean(d["time"]))
        a["deletion_auc_mean"] = float(np.mean(d["del"]))
        a["insertion_auc_mean"] = float(np.mean(d["ins"]))
        a["n"] = len(d["scores"])
        agg[name] = a

    print("\n%-22s %12s %9s %9s %9s" %
          ("method", "precision@k", "pointing", "mass", "time(s)"))
    for name, v in agg.items():
        print("%-22s %12.3f %9.3f %9.3f %9.2f"
              % (name, v["precision_at_k_mean"], v["pointing_mean"],
                 v["mass_on_event_mean"], v["time_s_mean"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"data": args.data, "head_accuracy": acc,
               "event_len": meta["event_len"], "n_explained": len(pos),
               "block": args.block, "aggregate": agg},
              open(args.out, "w"), indent=2)
    print("\nwrote", args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--root", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--backbone", default="mobilenetv2")
    b.add_argument("--backbone-weights", default=None)
    b.add_argument("--n", type=int, default=600)
    b.add_argument("--frames", type=int, default=40)
    b.add_argument("--input-size", type=int, default=128)
    b.add_argument("--event-len", type=int, default=5)
    b.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--out", default="results/semisynthetic_xai.json")
    s.add_argument("--block", type=int, default=5)
    s.add_argument("--explain", type=int, default=60)
    s.add_argument("--shapley-samples", type=int, default=40)
    s.add_argument("--epochs", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    (build if args.cmd == "build" else score)(args)


if __name__ == "__main__":
    main()

"""Sanity benchmark for the temporal explanation methods.

Real video gives you no ground truth about which frames a model *should* rely
on, so there is no way to say whether an attribution is right - only whether
it is self-consistent. Here the ground truth is known by construction: clips
are sequences of random descriptors in which a short contiguous "event" has
been planted, and the label is whether the event is present. A Bi-LSTM head
trained on this reaches near-perfect accuracy, and a good temporal explainer
should put its mass on the planted frames and nowhere else.

Feature dimension and sequence length match the real pipeline (40 x 1280), so
the runtimes are comparable to the video case.

    python scripts/run_synthetic_xai_benchmark.py --out results/synth_xai.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_dataset(n=600, n_frames=40, dim=1280, event_len=5, snr=1.0, seed=0):
    """Half the clips contain a planted event, half do not."""
    rng = np.random.default_rng(seed)

    # the event is a fixed random direction in feature space, the same for
    # every positive clip, which is what makes it learnable
    signature = rng.normal(0, 1, dim).astype("float32")
    signature /= np.linalg.norm(signature)

    X = rng.normal(0, 1, (n, n_frames, dim)).astype("float32") * 0.1
    y = np.zeros(n, dtype="int64")
    gt = np.zeros((n, n_frames), dtype="float32")

    for i in range(n):
        if i % 2 == 0:
            continue
        y[i] = 1
        start = rng.integers(0, n_frames - event_len + 1)
        X[i, start:start + event_len] += snr * signature
        gt[i, start:start + event_len] = 1.0

    return X, y, gt, signature


def train_head(X, y, n_frames, dim, seed=0, epochs=30):
    import tensorflow as tf
    from tensorflow import keras
    from src.models.build import build_temporal_head

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    n_tr = int(0.7 * len(X))
    model = build_temporal_head(n_frames, dim, lstm_units=(64, 32))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    Y = keras.utils.to_categorical(y, 2)
    model.fit(X[:n_tr], Y[:n_tr], validation_split=0.15, epochs=epochs,
              batch_size=32, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=6, restore_best_weights=True)])
    acc = model.evaluate(X[n_tr:], Y[n_tr:], verbose=0)[1]
    return model, float(acc), n_tr


def score_attribution(attr, gt_mask, event_len):
    """How well does the attribution pick out the planted frames."""
    from scipy.stats import spearmanr

    attr = np.asarray(attr, dtype="float64")
    k = int(gt_mask.sum())
    if k == 0:
        return None

    top = np.argsort(attr)[::-1][:k]
    hit = float(gt_mask[top].sum() / k)                 # precision@k
    peak = float(gt_mask[int(np.argmax(attr))])         # pointing game
    rho = spearmanr(attr, gt_mask).correlation

    # mass of the (non-negative) attribution falling on the true event
    pos = np.clip(attr, 0, None)
    mass = float(pos[gt_mask > 0].sum() / max(pos.sum(), 1e-9))
    return {"precision_at_k": hit, "pointing": peak,
            "spearman_gt": float(rho) if rho == rho else 0.0,
            "mass_on_event": mass}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--dim", type=int, default=1280)
    ap.add_argument("--event-len", type=int, default=5)
    ap.add_argument("--snr", type=float, default=1.0)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--explain", type=int, default=60,
                    help="number of positive test clips to explain")
    ap.add_argument("--shapley-samples", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/synth_xai.json")
    args = ap.parse_args()

    from src.xai import temporal_baselines as tb
    from src.xai.frame_importance import importance_attribution
    from src.xai.metrics import (temporal_deletion_curve,
                                 temporal_insertion_curve, auc)

    all_seeds = []
    for seed in range(args.seeds):
        print("\n=== seed %d ===" % seed)
        X, y, gt, _ = make_dataset(args.n, args.frames, args.dim,
                                   args.event_len, args.snr, seed=seed)
        head, acc, n_tr = train_head(X, y, args.frames, args.dim, seed=seed)
        print("head test accuracy: %.4f" % acc)

        # explain positive clips from the held-out part
        pos = [i for i in range(n_tr, len(X)) if y[i] == 1][:args.explain]
        print("explaining %d positive clips" % len(pos))

        methods = {
            "block_ablation": lambda f: importance_attribution(
                f, head, block=args.block),
            "occlusion": lambda f: tb.temporal_occlusion(f, head),
            "gradient_input": lambda f: tb.temporal_gradient_input(f, head),
            "integrated_gradients": lambda f: tb.temporal_integrated_gradients(
                f, head),
            "shapley_sampling": lambda f: tb.temporal_shapley_sampling(
                f, head, n_samples=args.shapley_samples, group=args.block),
        }

        acc_rows = {m: {"scores": [], "time": [], "del": [], "ins": []}
                    for m in methods}
        acc_rows["random"] = {"scores": [], "time": [], "del": [], "ins": []}
        rng = np.random.default_rng(seed)

        for n_done, i in enumerate(pos):
            feats, mask = X[i], gt[i]
            for name, fn in methods.items():
                t0 = time.perf_counter()
                attr = fn(feats)
                dt = time.perf_counter() - t0
                s = score_attribution(attr, mask, args.event_len)
                if s:
                    acc_rows[name]["scores"].append(s)
                    acc_rows[name]["time"].append(dt)
                    acc_rows[name]["del"].append(
                        auc(temporal_deletion_curve(feats, head, attr)))
                    acc_rows[name]["ins"].append(
                        auc(temporal_insertion_curve(feats, head, attr)))

            attr = rng.normal(size=args.frames)
            s = score_attribution(attr, mask, args.event_len)
            acc_rows["random"]["scores"].append(s)
            acc_rows["random"]["time"].append(0.0)
            acc_rows["random"]["del"].append(
                auc(temporal_deletion_curve(feats, head, attr)))
            acc_rows["random"]["ins"].append(
                auc(temporal_insertion_curve(feats, head, attr)))

            if (n_done + 1) % 10 == 0:
                print("  %d/%d" % (n_done + 1, len(pos)), flush=True)

        seed_out = {"seed": seed, "head_accuracy": acc, "methods": {}}
        for name, d in acc_rows.items():
            if not d["scores"]:
                continue
            keys = d["scores"][0].keys()
            seed_out["methods"][name] = {
                k: float(np.mean([s[k] for s in d["scores"]])) for k in keys}
            seed_out["methods"][name].update({
                "time_s": float(np.mean(d["time"])),
                "deletion_auc": float(np.mean(d["del"])),
                "insertion_auc": float(np.mean(d["ins"])),
            })
        all_seeds.append(seed_out)
        for k, v in seed_out["methods"].items():
            print("  %-22s P@k=%.3f point=%.3f mass=%.3f del=%.3f ins=%.3f %.3fs"
                  % (k, v["precision_at_k"], v["pointing"], v["mass_on_event"],
                     v["deletion_auc"], v["insertion_auc"], v["time_s"]))

    # aggregate across seeds
    names = all_seeds[0]["methods"].keys()
    agg = {}
    for name in names:
        vals = {}
        for k in all_seeds[0]["methods"][name]:
            arr = [s["methods"][name][k] for s in all_seeds]
            vals[k + "_mean"] = float(np.mean(arr))
            vals[k + "_std"] = float(np.std(arr, ddof=1) if len(arr) > 1 else 0)
        agg[name] = vals

    out = {"config": vars(args), "per_seed": all_seeds, "aggregate": agg,
           "head_accuracy_mean": float(np.mean(
               [s["head_accuracy"] for s in all_seeds]))}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print("\n%-22s %14s %12s %12s %10s %10s" %
          ("method", "precision@k", "pointing", "mass", "del AUC", "time(s)"))
    for name, v in agg.items():
        print("%-22s %6.3f+-%.3f %6.3f %11.3f %10.3f %10.3f" % (
            name, v["precision_at_k_mean"], v["precision_at_k_std"],
            v["pointing_mean"], v["mass_on_event_mean"],
            v["deletion_auc_mean"], v["time_s_mean"]))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

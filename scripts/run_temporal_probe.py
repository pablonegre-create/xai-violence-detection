"""Does the model use temporal order at all?

The component ablation found that replacing the Bi-LSTM with mean pooling over
the same frame descriptors costs nothing. Two explanations fit: either the
recurrent layer is redundant because the task is solved from per-frame
appearance, or it is undertrained and the comparison is uninformative.

This settles it directly. A model that uses temporal order must degrade when
the order is destroyed at test time. Three destructive transforms are applied
to the cached descriptors of the test split:

  shuffle   frames permuted at random - removes order, keeps the multiset
  reverse   frames played backwards - removes direction, keeps adjacency
  sort      frames ordered by descriptor norm - removes order deterministically

Unchanged accuracy is necessary but not sufficient: accuracy is an aggregate,
and flips in both directions can cancel. So the probe also reports what happens
to the decision function itself - how far the violent-class probability moves
per clip, how often the predicted label changes, and whether either grows once
the permutation is repeated. Shuffle is run --n-perm times with different
permutations; reverse and sort are deterministic and run once.

    python scripts/run_temporal_probe.py --features features/rlvs \\
        --head-weights ckpt/head_rlvs_seed0.h5 --out results/temporal_probe.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def shuffled(X, rng):
    """Permute each clip's frames independently."""
    return np.stack([x[rng.permutation(len(x))] for x in X])


def sorted_by_norm(X):
    return np.stack([x[np.argsort(np.linalg.norm(x, axis=1))] for x in X])


def describe(p_ref, pred_ref, p_new, yte):
    """What the transform did to the decision function, not just to accuracy."""
    from sklearn.metrics import accuracy_score, f1_score
    pred = p_new.argmax(1)
    dp = np.abs(p_new[:, 1] - p_ref[:, 1])
    flips = pred != pred_ref
    # a correlation is undefined if either side is constant
    if p_ref[:, 1].std() < 1e-12 or p_new[:, 1].std() < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(p_ref[:, 1], p_new[:, 1])[0, 1])
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
        "flip_rate": float(flips.mean()),
        "n_flipped": int(flips.sum()),
        "abs_dp_mean": float(dp.mean()),
        "abs_dp_median": float(np.median(dp)),
        "abs_dp_p95": float(np.percentile(dp, 95)),
        "abs_dp_max": float(dp.max()),
        "prob_corr": corr,
    }


def aggregate(runs):
    """mean and sd over repeated permutations of the same transform."""
    keys = [k for k in runs[0] if isinstance(runs[0][k], float)]
    out = {}
    for k in keys:
        v = np.asarray([r[k] for r in runs], float)
        out[k] = float(np.nanmean(v))
        out[k + "_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
    out["n_runs"] = len(runs)
    out["n_flipped_max"] = int(max(r["n_flipped"] for r in runs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--head-weights", required=True)
    ap.add_argument("--lstm", nargs="+", type=int, default=[64, 32])
    ap.add_argument("--head", default="bilstm",
                    choices=["bilstm", "bigru", "avgpool"])
    ap.add_argument("--n-perm", type=int, default=20,
                    help="independent random permutations of the frame axis")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/temporal_probe.json")
    args = ap.parse_args()

    from src.models.build import (build_temporal_head, build_gru_head,
                                  build_pooling_head)
    from scripts.train import load_split

    X, y, split, _ = load_split(args.features)
    te = split == "test"
    Xte, yte = X[te], y[te]
    n, n_frames, dim = Xte.shape
    print("test split: %d clips, %d frames, %d dims" % (n, n_frames, dim))

    if args.head == "bilstm":
        head = build_temporal_head(n_frames, dim, lstm_units=tuple(args.lstm))
    elif args.head == "bigru":
        head = build_gru_head(n_frames, dim, units=tuple(args.lstm))
    else:
        head = build_pooling_head(n_frames, dim)
    head.load_weights(args.head_weights)

    def predict(A):
        return head.predict(A.astype("float32"), batch_size=64, verbose=0)

    p_ref = predict(Xte)
    pred_ref = p_ref.argmax(1)
    base = float((pred_ref == yte).mean())
    print("untransformed accuracy %.4f" % base)

    rng = np.random.default_rng(args.seed)
    transforms = {}

    runs = [describe(p_ref, pred_ref, predict(shuffled(Xte, rng)), yte)
            for _ in range(args.n_perm)]
    transforms["shuffle"] = aggregate(runs)
    transforms["shuffle"]["per_permutation_accuracy"] = [r["accuracy"]
                                                         for r in runs]

    transforms["reverse"] = describe(p_ref, pred_ref, predict(Xte[:, ::-1]),
                                     yte)
    transforms["sort_by_norm"] = describe(p_ref, pred_ref,
                                          predict(sorted_by_norm(Xte)), yte)
    transforms["identity"] = describe(p_ref, pred_ref, p_ref, yte)

    for k, v in transforms.items():
        v["delta"] = v["accuracy"] - base

    print()
    print("%-14s %8s %8s %9s %9s %9s" %
          ("transform", "acc", "delta", "flips", "|dp| mean", "|dp| max"))
    for k in ("identity", "shuffle", "reverse", "sort_by_norm"):
        v = transforms[k]
        print("%-14s %8.4f %+8.4f %8.1f%% %9.4f %9.4f" %
              (k, v["accuracy"], v["delta"], 100 * v["flip_rate"],
               v["abs_dp_mean"], v["abs_dp_max"]))

    # A fixed threshold is wrong here: on a 50-clip test split a single clip is
    # already 2 points. Compare each drop against the binomial noise of this
    # split instead, so the verdict means the same thing at every sample size.
    noise = 1.96 * np.sqrt(base * (1 - base) / max(n, 1))
    worst = min(v["delta"] for v in transforms.values())
    significant = worst < -noise

    verdict = ("order-sensitive: destroying the sequence costs more accuracy "
               "than sampling noise"
               if significant else
               "order-invariant: no transform costs more than sampling noise")
    print("\nlargest drop %+.4f, noise floor +/-%.4f (%d clips, 1 clip = %.4f)"
          % (worst, noise, n, 1.0 / max(n, 1)))
    print("-> %s" % verdict)

    # the function-level summary the accuracy number cannot give you
    sh = transforms["shuffle"]
    print("shuffle over %d permutations: flip rate %.2f%% (max %d clips), "
          "mean |dp| %.4f, p95 |dp| %.4f"
          % (sh["n_runs"], 100 * sh["flip_rate"], sh["n_flipped_max"],
             sh["abs_dp_mean"], sh["abs_dp_p95"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"features": args.features, "head": args.head,
               "n_test": n, "n_perm": args.n_perm,
               "base_accuracy": base, "transforms": transforms,
               "largest_drop": worst, "noise_floor_95": float(noise),
               "clips_per_point": 1.0 / max(n, 1),
               "significant": bool(significant), "verdict": verdict},
              open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

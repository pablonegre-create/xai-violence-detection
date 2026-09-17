"""Does the model use temporal order at all?

The component ablation found that replacing the Bi-LSTM with mean pooling over
the same frame descriptors costs nothing (96.15% against 95.30%, p = 1.0). Two
explanations fit: either the recurrent layer is redundant because the task is
solved from per-frame appearance, or it is undertrained and the comparison is
uninformative.

This settles it directly. A model that uses temporal order must degrade when
the order is destroyed at test time. Three destructive transforms are applied
to the cached descriptors of the test split:

  shuffle   frames permuted at random - removes order, keeps the multiset
  reverse   frames played backwards - removes direction, keeps adjacency
  sort      frames ordered by descriptor norm - removes order deterministically

If accuracy survives all three, the decision does not depend on the sequence,
and any temporal explanation of it is explaining something the model is not
doing. The synthetic benchmark is the positive control: there the same probe
must show a large drop.

    python scripts/run_temporal_probe.py --features features/rlvs \\
        --head-weights ckpt/head_rlvs_seed0.h5 --out results/temporal_probe.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def transforms(rng):
    return {
        "identity": lambda X: X,
        "shuffle": lambda X: np.stack([x[rng.permutation(len(x))] for x in X]),
        "reverse": lambda X: X[:, ::-1],
        "sort_by_norm": lambda X: np.stack(
            [x[np.argsort(np.linalg.norm(x, axis=1))] for x in X]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--head-weights", required=True)
    ap.add_argument("--lstm", nargs="+", type=int, default=[64, 32])
    ap.add_argument("--head", default="bilstm",
                    choices=["bilstm", "bigru", "avgpool"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/temporal_probe.json")
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score, f1_score
    from src.models.build import (build_temporal_head, build_gru_head,
                                  build_pooling_head)
    from scripts.train import load_split

    X, y, split, _ = load_split(args.features)
    te = split == "test"
    Xte, yte = X[te], y[te]
    print("test split: %d clips, %d frames, %d dims"
          % (len(Xte), Xte.shape[1], Xte.shape[2]))

    if args.head == "bilstm":
        head = build_temporal_head(X.shape[1], X.shape[2],
                                   lstm_units=tuple(args.lstm))
    elif args.head == "bigru":
        head = build_gru_head(X.shape[1], X.shape[2], units=tuple(args.lstm))
    else:
        head = build_pooling_head(X.shape[1], X.shape[2])
    head.load_weights(args.head_weights)

    rng = np.random.default_rng(args.seed)
    rows = {}
    base = None
    for name, fn in transforms(rng).items():
        prob = head.predict(fn(Xte).astype("float32"), batch_size=64, verbose=0)
        pred = prob.argmax(1)
        acc = float(accuracy_score(yte, pred))
        f1 = float(f1_score(yte, pred, zero_division=0))
        if base is None:
            base = acc
        rows[name] = {"accuracy": acc, "f1": f1, "delta": acc - base}
        print("  %-14s acc %.4f  f1 %.4f  delta %+.4f"
              % (name, acc, f1, acc - base))

    # A fixed threshold is wrong here: on a 50-clip test split a single clip is
    # already 2 points. Compare each drop against the binomial noise of this
    # split instead, so the verdict means the same thing at every sample size.
    n = int(te.sum())
    noise = 1.96 * np.sqrt(base * (1 - base) / max(n, 1))
    worst = min(r["delta"] for r in rows.values())
    significant = worst < -noise

    verdict = ("order-sensitive: destroying the sequence costs more accuracy "
               "than sampling noise"
               if significant else
               "order-invariant: no transform costs more than sampling noise")
    print("\nlargest drop %+.4f, noise floor +/-%.4f (%d clips, 1 clip = %.4f)"
          % (worst, noise, n, 1.0 / max(n, 1)))
    print("-> %s" % verdict)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"features": args.features, "head": args.head,
               "n_test": n, "transforms": rows, "largest_drop": worst,
               "noise_floor_95": float(noise),
               "clips_per_point": 1.0 / max(n, 1),
               "significant": bool(significant), "verdict": verdict},
              open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

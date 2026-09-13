"""Cross-dataset generalisation.

Train the head on one dataset's cached features and test it on every other
dataset's, without any adaptation. The backbone is the same throughout, so
what is being measured is how much of the temporal decision rule transfers.

Needs one feature cache per dataset, all produced with the same backbone
checkpoint:

    for d in rlvs hockey movies violentflows rwf2000; do
      python scripts/extract_features.py --dataset $d --root $DATA/$d \
          --out features/$d --backbone-weights ckpt/mnv2_ft5.h5
    done
    python scripts/run_cross_dataset.py --features features --out results/cross.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATASETS = ["rlvs", "rwf2000", "hockey", "movies", "violentflows"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True,
                    help="directory containing one subdir per dataset")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="results/cross_dataset.json")
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from src.models.build import build_temporal_head
    from scripts.train import load_split

    data = {}
    for d in args.datasets:
        p = os.path.join(args.features, d)
        if not os.path.isdir(p):
            print("missing feature cache for %s, skipping" % d)
            continue
        X, y, s, meta = load_split(p)
        data[d] = (X, y, s)
        print("%-14s %5d clips, %d frames x %d dims" %
              (d, len(X), X.shape[1], X.shape[2]))

    names = list(data)
    matrix = {src: {} for src in names}

    for src in names:
        Xs, ys, ss = data[src]
        tr = ss != "test" if (ss == "test").any() else np.ones(len(Xs), bool)

        per_seed = {tgt: [] for tgt in names}
        for seed in range(args.seeds):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)

            head = build_temporal_head(Xs.shape[1], Xs.shape[2])
            head.compile(optimizer=keras.optimizers.Adam(1e-3),
                         loss="categorical_crossentropy", metrics=["accuracy"])
            head.fit(Xs[tr], keras.utils.to_categorical(ys[tr], 2),
                     validation_split=0.12, epochs=args.epochs, batch_size=16,
                     verbose=0,
                     callbacks=[keras.callbacks.EarlyStopping(
                         monitor="val_loss", patience=10,
                         restore_best_weights=True)])

            for tgt in names:
                Xt, yt, st = data[tgt]
                if tgt == src:
                    mask = st == "test"
                    if not mask.any():
                        mask = np.ones(len(Xt), bool)
                else:
                    mask = np.ones(len(Xt), bool)   # whole target set

                prob = head.predict(Xt[mask], batch_size=64, verbose=0)
                pred = prob.argmax(1)
                per_seed[tgt].append({
                    "accuracy": float(accuracy_score(yt[mask], pred)),
                    "f1": float(f1_score(yt[mask], pred, zero_division=0)),
                    "auc": float(roc_auc_score(yt[mask], prob[:, 1]))
                    if len(set(yt[mask])) > 1 else float("nan"),
                })

        for tgt in names:
            accs = [r["accuracy"] for r in per_seed[tgt]]
            f1s = [r["f1"] for r in per_seed[tgt]]
            aucs = [r["auc"] for r in per_seed[tgt]]
            matrix[src][tgt] = {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs, ddof=1) if len(accs) > 1 else 0),
                "f1_mean": float(np.mean(f1s)),
                "auc_mean": float(np.nanmean(aucs)),
                "in_domain": src == tgt,
            }
        print("\ntrained on %s:" % src)
        for tgt in names:
            m = matrix[src][tgt]
            print("   -> %-14s acc %.4f +/- %.4f %s" %
                  (tgt, m["accuracy_mean"], m["accuracy_std"],
                   "(in-domain)" if m["in_domain"] else ""))

    json.dump({"datasets": names, "seeds": args.seeds, "matrix": matrix},
              open(args.out, "w"), indent=2)

    print("\n%-14s" % "train\\test", end="")
    for t in names:
        print("%12s" % t, end="")
    print()
    for s in names:
        print("%-14s" % s, end="")
        for t in names:
            print("%12.3f" % matrix[s][t]["accuracy_mean"], end="")
        print()
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

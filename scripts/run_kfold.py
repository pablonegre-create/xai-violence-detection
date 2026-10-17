"""k-fold cross-validation on the cached descriptors.

The paper reports a stratified 70/10/20 holdout repeated over seeds. A reviewer
pointed out that on datasets this small a single holdout can flatter or punish
a model depending on which clips land in the test split, and asked for 10-fold
instead. The descriptors do not depend on the split, so the folds are cut over
the cached arrays and nothing is re-extracted.

Folds respect duplicate groups, for the same reason the holdout does: the
distributed mirrors contain byte-identical clips under different names, and a
copy in the training folds makes the test fold easier than it should be.

    python scripts/run_kfold.py --features features/rlvs --k 10 \\
        --out results/kfold_rlvs.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="results/kfold.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lstm", nargs="+", type=int, default=[64, 32])
    ap.add_argument("--dense", nargs="+", type=int, default=[64, 32])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.12,
                    help="carved out of the training folds for early stopping")
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 roc_auc_score)
    from scripts.train import load_split
    from src.data.datasets import duplicate_groups
    from src.models.build import build_temporal_head
    from src.eval.stats import summarise_runs

    X, y, split, meta = load_split(args.features)
    n = len(y)
    name = args.dataset or os.path.basename(os.path.normpath(args.features))
    print("%s: %d clips, features %s" % (name, n, X.shape))

    # group ids keep identical clips in one fold. The feature cache lists the
    # clip file names, so the groups are recovered from those rather than by
    # hashing the videos again.
    names = meta.get("names")
    gid = np.arange(n)
    if names:
        try:
            items = [(p, int(l)) for p, l in zip(names, y)]
            gid = np.asarray(duplicate_groups(items))
            n_groups = len(set(gid.tolist()))
            if n_groups < n:
                print("%d clips fall into %d duplicate groups"
                      % (n, n_groups))
        except Exception as e:
            # the cache stores relative names; if the files are not next to us
            # the grouping cannot be recomputed and every clip is its own group
            print("duplicate grouping unavailable (%s); folds cut per clip" % e)

    rng = np.random.default_rng(args.seed)
    groups = np.unique(gid)
    # stratify the folds by the group's majority label so every fold keeps the
    # class balance, then interleave so fold sizes stay even
    glabel = np.asarray([int(round(y[gid == g].mean())) for g in groups])
    fold_of_group = np.empty(len(groups), int)
    for cls in (0, 1):
        idx = np.where(glabel == cls)[0]
        rng.shuffle(idx)
        fold_of_group[idx] = np.arange(len(idx)) % args.k
    fold = np.empty(n, int)
    for g, f in zip(groups, fold_of_group):
        fold[gid == g] = f

    Y = keras.utils.to_categorical(y, 2)
    runs = []
    for k in range(args.k):
        te = fold == k
        pool = np.where(~te)[0]
        rng.shuffle(pool)
        n_val = max(1, int(round(args.val_fraction * len(pool))))
        va_idx, tr_idx = pool[:n_val], pool[n_val:]

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed + k)
        model = build_temporal_head(X.shape[1], X.shape[2],
                                    lstm_units=tuple(args.lstm),
                                    dense_units=tuple(args.dense))
        model.compile(optimizer=keras.optimizers.Adam(args.lr),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(X[tr_idx], Y[tr_idx], validation_data=(X[va_idx], Y[va_idx]),
                  epochs=args.epochs, batch_size=args.batch, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(
                                 monitor="val_loss", patience=10,
                                 restore_best_weights=True),
                             keras.callbacks.ReduceLROnPlateau(
                                 monitor="val_loss", factor=0.5, patience=5,
                                 min_lr=1e-6)])

        prob = model.predict(X[te], batch_size=64, verbose=0)
        pred = prob.argmax(1)
        p, r, f1, _ = precision_recall_fscore_support(y[te], pred,
                                                      average="binary",
                                                      zero_division=0)
        try:
            auc = float(roc_auc_score(y[te], prob[:, 1]))
        except ValueError:
            auc = float("nan")
        runs.append({"fold": k, "n_test": int(te.sum()),
                     "accuracy": float(accuracy_score(y[te], pred)),
                     "precision": float(p), "recall": float(r),
                     "f1": float(f1), "auc": auc})
        print("fold %2d/%d  n=%4d  acc=%.4f  f1=%.4f  auc=%.4f"
              % (k + 1, args.k, te.sum(), runs[-1]["accuracy"], f1, auc),
              flush=True)

    accs = [r["accuracy"] for r in runs]
    summary = {
        "dataset": name, "features": args.features, "k": args.k,
        "accuracy": summarise_runs(accs, n_test=n),
        "f1": summarise_runs([r["f1"] for r in runs]),
        "precision": summarise_runs([r["precision"] for r in runs]),
        "recall": summarise_runs([r["recall"] for r in runs]),
        "auc": summarise_runs([r["auc"] for r in runs]),
        "folds": runs,
    }
    print("\n%d-fold accuracy %.4f +/- %.4f   (min %.4f, max %.4f)"
          % (args.k, summary["accuracy"]["mean"], summary["accuracy"]["std"],
             summary["accuracy"]["min"], summary["accuracy"]["max"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

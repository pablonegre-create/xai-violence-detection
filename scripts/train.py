"""Train the recurrent head on cached descriptors, over several seeds.

Stage 1 (fine-tuning the CNN) is scripts/finetune_backbone.py; this is stage 2.
Repeating over seeds is what produces the mean +/- sd and the CIs the reviewers
asked for, so --seeds defaults to 5.

    python scripts/train.py --features features/rlvs --seeds 5 \
        --out results/rlvs_main.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_split(feat_dir):
    X = np.load(os.path.join(feat_dir, "features.npy"))
    y = np.load(os.path.join(feat_dir, "labels.npy"))
    meta = json.load(open(os.path.join(feat_dir, "index.json")))
    s = np.asarray(meta["splits"])
    return X, y, s, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="results/train.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lstm", nargs="+", type=int, default=[64, 32])
    ap.add_argument("--dense", nargs="+", type=int, default=[64, 32])
    ap.add_argument("--batch-norm", action="store_true")
    ap.add_argument("--head", default="bilstm",
                    choices=["bilstm", "bigru", "avgpool"])
    ap.add_argument("--save-preds", default=None)
    ap.add_argument("--ckpt-prefix", default=None,
                    help="save head weights as <prefix>_seed<N>.h5; the XAI and "
                         "robustness scripts need these")
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 roc_auc_score, confusion_matrix)
    from src.models.build import (build_temporal_head, build_gru_head,
                                  build_pooling_head)
    from src.eval.stats import summarise_runs, wilson_interval

    X, y, split, meta = load_split(args.features)
    n_frames, dim = X.shape[1], X.shape[2]

    tr, va, te = split == "train", split == "val", split == "test"
    if va.sum() == 0:                      # datasets with no val split
        va = tr.copy()
    print("train/val/test = %d/%d/%d, features %s" %
          (tr.sum(), va.sum(), te.sum(), X.shape))

    Y = keras.utils.to_categorical(y, 2)
    runs, all_preds = [], {}

    for seed in range(args.seeds):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)

        if args.head == "bilstm":
            model = build_temporal_head(n_frames, dim,
                                        lstm_units=tuple(args.lstm),
                                        dense_units=tuple(args.dense),
                                        batch_norm=args.batch_norm)
        elif args.head == "bigru":
            model = build_gru_head(n_frames, dim, units=tuple(args.lstm))
        else:
            model = build_pooling_head(n_frames, dim)

        model.compile(optimizer=keras.optimizers.Adam(args.lr),
                      loss="categorical_crossentropy", metrics=["accuracy"])

        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=5, min_lr=1e-6),
        ]
        model.fit(X[tr], Y[tr], validation_data=(X[va], Y[va]),
                  epochs=args.epochs, batch_size=args.batch,
                  callbacks=cbs, verbose=0)

        if args.ckpt_prefix:
            os.makedirs(os.path.dirname(args.ckpt_prefix) or ".", exist_ok=True)
            w = "%s_seed%d.h5" % (args.ckpt_prefix, seed)
            model.save_weights(w)
            print("  saved", w)

        prob = model.predict(X[te], batch_size=64, verbose=0)
        pred = prob.argmax(1)
        acc = accuracy_score(y[te], pred)
        p, r, f1, _ = precision_recall_fscore_support(y[te], pred,
                                                      average="binary",
                                                      zero_division=0)
        try:
            auc = roc_auc_score(y[te], prob[:, 1])
        except ValueError:
            auc = float("nan")

        runs.append({"seed": seed, "accuracy": float(acc),
                     "precision": float(p), "recall": float(r),
                     "f1": float(f1), "auc": float(auc),
                     "confusion": confusion_matrix(y[te], pred).tolist()})
        all_preds["seed%d" % seed] = pred.tolist()
        print("seed %d  acc=%.4f  f1=%.4f  auc=%.4f" % (seed, acc, f1, auc),
              flush=True)

    accs = [r["accuracy"] for r in runs]
    summary = {
        "features": args.features,
        "head": args.head,
        "config": {"lstm": args.lstm, "dense": args.dense,
                   "batch_norm": args.batch_norm, "lr": args.lr,
                   "epochs": args.epochs, "batch": args.batch},
        "n_test": int(te.sum()),
        "accuracy": summarise_runs(accs, n_test=int(te.sum())),
        "f1": summarise_runs([r["f1"] for r in runs]),
        "precision": summarise_runs([r["precision"] for r in runs]),
        "recall": summarise_runs([r["recall"] for r in runs]),
        "auc": summarise_runs([r["auc"] for r in runs]),
        "runs": runs,
    }
    print("\naccuracy %.4f +/- %.4f  (95%% CI %.4f-%.4f)" % (
        summary["accuracy"]["mean"], summary["accuracy"]["std"],
        *summary["accuracy"]["wilson95"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    if args.save_preds:
        json.dump({"y_true": y[te].tolist(), "preds": all_preds},
                  open(args.save_preds, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

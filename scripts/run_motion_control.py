"""Is the RWF-2000 gap really about missing motion?

Showing that a model ignores frame order does not by itself prove that the
points it loses on RWF-2000 are the motion it cannot see. Something else about
that dataset could be responsible. The control is to keep the whole pipeline
fixed and change only whether motion is present in the input:

  rgb    frames as they are, which is the detector in the paper
  diff   |frame_t - frame_{t-1}| through the same backbone: motion, little
         appearance
  both   the two descriptor sets concatenated

Same backbone architecture, same recurrent head, same split, same seeds, same
schedule. The only difference is what the backbone was shown. If `diff` and
`both` recover a large part of the gap on RWF-2000 and change little on a
dataset that is already separable from appearance, the gap is motion. If they
do not, it is not, and the paper has to say so.

The order probe is run on each arm as well. We expected it to stay flat on the
appearance arm and move on the motion arm. It stays flat on both: the frame
difference puts the motion inside each per-frame descriptor, so a head that
ignores order can still use it. That is reported in the paper as it came out.

    python scripts/run_motion_control.py --dataset rwf2000 \\
        --features-rgb features/rwf2000 --features-diff features/rwf2000_diff \\
        --out results/motion_control_rwf2000.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def fit_and_score(X, y, split, seed, epochs, batch, lr, lstm, dense):
    """One seed: train the standard head on these descriptors, score the test
    split, and return the test probabilities so the probe can reuse them."""
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 roc_auc_score)
    from src.models.build import build_temporal_head

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    tr, va, te = split == "train", split == "val", split == "test"
    if va.sum() == 0:
        va = tr.copy()

    Y = keras.utils.to_categorical(y, 2)
    model = build_temporal_head(X.shape[1], X.shape[2],
                                lstm_units=tuple(lstm),
                                dense_units=tuple(dense))
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X[tr], Y[tr], validation_data=(X[va], Y[va]),
              epochs=epochs, batch_size=batch, verbose=0,
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
    return {"seed": seed,
            "accuracy": float(accuracy_score(y[te], pred)),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "auc": auc}, model, prob


def probe(model, X, y, split, n_perm, seed):
    """Shuffle the frame axis of the test split and see what moves."""
    te = split == "test"
    Xte, yte = X[te], y[te]
    p_ref = model.predict(Xte.astype("float32"), batch_size=64, verbose=0)
    pred_ref = p_ref.argmax(1)
    base = float((pred_ref == yte).mean())

    rng = np.random.default_rng(seed)
    accs, flips, dps = [], [], []
    for _ in range(n_perm):
        Xp = np.stack([x[rng.permutation(len(x))] for x in Xte])
        p = model.predict(Xp.astype("float32"), batch_size=64, verbose=0)
        pred = p.argmax(1)
        accs.append(float((pred == yte).mean()))
        flips.append(float((pred != pred_ref).mean()))
        dps.append(float(np.abs(p[:, 1] - p_ref[:, 1]).mean()))

    noise = 1.96 * np.sqrt(base * (1 - base) / max(len(yte), 1))
    return {"base_accuracy": base,
            "shuffle_accuracy_mean": float(np.mean(accs)),
            "shuffle_accuracy_sd": float(np.std(accs, ddof=1)) if n_perm > 1
            else 0.0,
            "delta": float(np.mean(accs)) - base,
            "flip_rate": float(np.mean(flips)),
            "abs_dp_mean": float(np.mean(dps)),
            "noise_floor_95": float(noise),
            "order_sensitive": bool(float(np.mean(accs)) - base < -noise)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--features-rgb", required=True)
    ap.add_argument("--features-diff", required=True)
    ap.add_argument("--out", default="results/motion_control.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lstm", nargs="+", type=int, default=[64, 32])
    ap.add_argument("--dense", nargs="+", type=int, default=[64, 32])
    ap.add_argument("--n-perm", type=int, default=10)
    args = ap.parse_args()

    from scripts.train import load_split
    from src.eval.stats import (summarise_runs, paired_ttest_seeds,
                                paired_diff_ci)

    Xr, yr, sr, _ = load_split(args.features_rgb)
    Xd, yd, sd, _ = load_split(args.features_diff)

    # the two caches must describe the same clips in the same order, or the
    # concatenated arm silently pairs one clip's appearance with another's
    # motion and the comparison is meaningless
    if Xr.shape[0] != Xd.shape[0] or not np.array_equal(yr, yd):
        raise SystemExit("rgb and diff caches disagree: %d vs %d clips"
                         % (Xr.shape[0], Xd.shape[0]))
    if not np.array_equal(sr, sd):
        raise SystemExit("rgb and diff caches have different splits")

    arms = {
        "appearance (rgb)": Xr,
        "motion (frame difference)": Xd,
        "both (concatenated)": np.concatenate([Xr, Xd], axis=2),
    }

    results = {}
    for name, X in arms.items():
        print("\n=== %s  features %s ===" % (name, X.shape), flush=True)
        runs, last_model = [], None
        for seed in range(args.seeds):
            r, model, _ = fit_and_score(X, yr, sr, seed, args.epochs,
                                        args.batch, args.lr, args.lstm,
                                        args.dense)
            runs.append(r)
            last_model = model
            print("  seed %d  acc=%.4f  f1=%.4f  auc=%.4f"
                  % (seed, r["accuracy"], r["f1"], r["auc"]), flush=True)

        accs = [r["accuracy"] for r in runs]
        results[name] = {
            "accuracy": summarise_runs(accs, n_test=int((sr == "test").sum())),
            "f1": summarise_runs([r["f1"] for r in runs]),
            "precision": summarise_runs([r["precision"] for r in runs]),
            "recall": summarise_runs([r["recall"] for r in runs]),
            "auc": summarise_runs([r["auc"] for r in runs]),
            "feature_dim": int(X.shape[2]),
            "runs": runs,
            # the probe on the last seed's model, which is enough to say
            # whether this arm depends on order at all
            "order_probe": probe(last_model, X, yr, sr, args.n_perm, 0),
        }

    ref = "appearance (rgb)"
    ref_accs = [r["accuracy"] for r in results[ref]["runs"]]
    for name in results:
        if name == ref:
            continue
        accs = [r["accuracy"] for r in results[name]["runs"]]
        results[name]["vs_appearance"] = {
            "delta": results[name]["accuracy"]["mean"]
            - results[ref]["accuracy"]["mean"],
            "ttest": paired_ttest_seeds(accs, ref_accs),
            "ci": paired_diff_ci(accs, ref_accs),
        }

    print("\n%-28s %8s %8s %9s %9s" % ("arm", "acc", "sd", "delta", "probe"))
    for name, r in results.items():
        d = r.get("vs_appearance", {}).get("delta")
        print("%-28s %8.4f %8.4f %9s %9s"
              % (name, r["accuracy"]["mean"], r["accuracy"]["std"],
                 "--" if d is None else "%+.4f" % d,
                 "sensitive" if r["order_probe"]["order_sensitive"]
                 else "invariant"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"dataset": args.dataset, "seeds": args.seeds,
               "reference": ref, "arms": results},
              open(args.out, "w"), indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

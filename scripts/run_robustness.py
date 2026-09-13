"""Robustness to corruptions.

The corruption is applied to the decoded clip, so the backbone has to be rerun
- unlike the other experiments this one cannot work from cached features. That
makes it the expensive script in the repo; --severities 3 --subset 300 is a
reasonable compromise on a single GPU.

    python scripts/run_robustness.py --root $DATA/RLVS \
        --backbone-weights ckpt/mnv2_ft5.h5 --head-weights ckpt/head_seed0.h5 \
        --out results/robustness_rlvs.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--backbone-weights", required=True)
    ap.add_argument("--head-weights", required=True)
    ap.add_argument("--backbone", default="mobilenetv2")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--corruptions", nargs="+", default=None)
    ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/robustness.json")
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score
    from src.data.datasets import index_flat, load_clip, stratified_split
    from src.models.build import build_backbone, build_temporal_head, build_classifier_head
    from src.robustness.corruptions import CORRUPTIONS, mean_corruption_error, fgsm

    items = index_flat(args.root)
    _, _, test = stratified_split(items, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    if args.subset > 0 and len(test) > args.subset:
        pick = rng.choice(len(test), args.subset, replace=False)
        test = [test[i] for i in pick]
    print("%d test clips" % len(test))

    cnn = build_backbone(args.backbone, input_size=args.input_size)
    cnn.load_weights(args.backbone_weights, by_name=True, skip_mismatch=True)
    head = build_temporal_head(args.frames, cnn.output_shape[-1])
    head.load_weights(args.head_weights)

    def predict(clips):
        out = []
        for c in clips:
            f = cnn.predict(c, batch_size=args.frames, verbose=0)
            out.append(head.predict(f[None], verbose=0)[0].argmax())
        return np.asarray(out)

    print("decoding clips...")
    clean, labels = [], []
    for p, y in test:
        try:
            clean.append(load_clip(p, n_frames=args.frames,
                                   size=args.input_size))
            labels.append(y)
        except Exception as e:
            print(" skip", p, e)
    labels = np.asarray(labels)

    acc_clean = accuracy_score(labels, predict(clean))
    print("clean accuracy: %.4f" % acc_clean)

    names = args.corruptions or list(CORRUPTIONS)
    rows = []
    for name in names:
        fn = CORRUPTIONS[name]
        for sev in args.severities:
            corrupted = [fn(c, sev) for c in clean]
            acc = accuracy_score(labels, predict(corrupted))
            rows.append({"corruption": name, "severity": sev,
                         "accuracy": float(acc),
                         "drop": float(acc_clean - acc),
                         "relative_error": mean_corruption_error(acc, acc_clean)})
            print("  %-20s sev %d  acc %.4f  (drop %.4f)" %
                  (name, sev, acc, acc_clean - acc), flush=True)

    out = {"clean_accuracy": float(acc_clean), "n_clips": len(clean),
           "rows": rows}

    if args.adversarial:
        print("\nFGSM on the frame classifier")
        frame_cls = build_classifier_head(cnn)
        adv_rows = []
        for eps in (1 / 255., 2 / 255., 4 / 255., 8 / 255.):
            adv = []
            for c, y in zip(clean, labels):
                adv.append(fgsm(c, frame_cls, int(y), eps=eps))
            acc = accuracy_score(labels, predict(adv))
            adv_rows.append({"eps_255": eps * 255, "accuracy": float(acc),
                             "drop": float(acc_clean - acc)})
            print("  eps=%.0f/255  acc %.4f" % (eps * 255, acc))
        out["adversarial_fgsm"] = adv_rows

    # one number for the abstract: mean accuracy across all corruptions at
    # middle severity
    mid = [r["accuracy"] for r in rows if r["severity"] == 3]
    if mid:
        out["mean_accuracy_severity3"] = float(np.mean(mid))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

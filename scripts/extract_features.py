"""Stage 1: run the (fine-tuned) backbone over every clip and cache the
per-frame descriptors to .npy.

Everything downstream - training the recurrent head, the ablation sweep, the
temporal XAI comparison - reads these arrays instead of decoding video again,
which is what makes the explanation sweep cheap.

    python scripts/extract_features.py --dataset rlvs \
        --root /path/to/RLVS --out features/rlvs --backbone-weights ckpt/mnv2.h5
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="mobilenetv2")
    ap.add_argument("--backbone-weights", default=None,
                    help="fine-tuned checkpoint; ImageNet weights if omitted")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from src.data.datasets import (index_flat, index_rwf2000, load_clip,
                                   stratified_split)
    from src.models.build import build_backbone

    os.makedirs(args.out, exist_ok=True)

    if args.dataset == "rwf2000":
        splits = index_rwf2000(args.root)
        items = splits["train"] + splits["test"]
        split_of = {p: "train" for p, _ in splits["train"]}
        split_of.update({p: "test" for p, _ in splits["test"]})
    else:
        items = index_flat(args.root)
        tr, va, te = stratified_split(items, seed=args.seed)
        # stratified_split keeps byte-identical clips in one partition; report
        # how many there were so the log records it
        from src.data.datasets import duplicate_groups
        gid = duplicate_groups(items)
        n_dup = len(items) - len(set(gid))
        if n_dup:
            print("%d duplicate clip(s) kept within a single split" % n_dup)
        split_of = {}
        for name, lst in (("train", tr), ("val", va), ("test", te)):
            split_of.update({p: name for p, _ in lst})

    print("%s: %d clips (%d violent)" %
          (args.dataset, len(items), sum(y for _, y in items)))

    cnn = build_backbone(args.backbone, input_size=args.input_size,
                         weights="imagenet")
    if args.backbone_weights:
        cnn.load_weights(args.backbone_weights, by_name=True, skip_mismatch=True)
        print("loaded", args.backbone_weights)

    feats, labels, names, splits_out = [], [], [], []
    for i, (path, label) in enumerate(items):
        try:
            clip = load_clip(path, n_frames=args.frames, size=args.input_size)
        except Exception as e:
            print("skip %s (%s)" % (path, e))
            continue
        f = cnn.predict(clip, batch_size=args.frames, verbose=0)
        feats.append(f.astype("float32"))
        labels.append(label)
        names.append(os.path.relpath(path, args.root))
        splits_out.append(split_of.get(path, "train"))
        if (i + 1) % 50 == 0:
            print("  %d/%d" % (i + 1, len(items)), flush=True)

    np.save(os.path.join(args.out, "features.npy"), np.stack(feats))
    np.save(os.path.join(args.out, "labels.npy"), np.asarray(labels))
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump({"names": names, "splits": splits_out,
                   "backbone": args.backbone, "frames": args.frames,
                   "input_size": args.input_size,
                   "weights": args.backbone_weights}, fh, indent=2)
    print("wrote %d feature arrays to %s" % (len(feats), args.out))


if __name__ == "__main__":
    main()

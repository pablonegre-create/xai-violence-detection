"""Stage 1: adapt the ImageNet backbone to violence frames.

Frames are treated as independent images and inherit the label of their clip,
which is noisy (not every frame of a violent clip shows violence) but is what
lets the CNN move away from ImageNet features. The number of unfrozen layers
is the knob the fine-tuning ablation sweeps.

    python scripts/finetune_backbone.py --root /data/RLVS --finetune 5 \
        --out ckpt/mnv2_ft5.h5
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def frame_generator(items, n_frames, size, batch, shuffle=True, seed=0):
    from src.data.datasets import load_clip
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))

    while True:
        if shuffle:
            rng.shuffle(idx)
        xs, ys = [], []
        for i in idx:
            path, label = items[i]
            try:
                clip = load_clip(path, n_frames=n_frames, size=size)
            except Exception:
                continue
            # a couple of frames per clip keeps batches diverse
            pick = rng.choice(len(clip), size=min(4, len(clip)), replace=False)
            for p in pick:
                xs.append(clip[p]); ys.append(label)
                if len(xs) == batch:
                    yield np.stack(xs), np.eye(2)[ys]
                    xs, ys = [], []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="mobilenetv2")
    ap.add_argument("--finetune", default="5",
                    help="number of trainable top layers, or 'all' or '0'")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    from src.data.datasets import index_flat, stratified_split
    from src.models.build import (build_backbone, build_classifier_head,
                                  set_finetune_depth)

    tf.keras.utils.set_random_seed(args.seed)

    items = index_flat(args.root)
    tr, va, _ = stratified_split(items, seed=args.seed)
    print("%d train clips, %d val clips" % (len(tr), len(va)))

    cnn = build_backbone(args.backbone, input_size=args.input_size)
    depth = "all" if args.finetune == "all" else int(args.finetune)
    set_finetune_depth(cnn, depth)
    model = build_classifier_head(cnn)

    trainable = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    print("finetune depth=%s -> %d trainable params" % (depth, trainable))

    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    gen_tr = frame_generator(tr, args.frames, args.input_size, args.batch,
                             seed=args.seed)
    gen_va = frame_generator(va, args.frames, args.input_size, args.batch,
                             shuffle=False, seed=args.seed)

    model.fit(gen_tr, steps_per_epoch=args.steps, epochs=args.epochs,
              validation_data=gen_va, validation_steps=40,
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=4, restore_best_weights=True)],
              verbose=2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cnn.save_weights(args.out)

    # The frame classifier is kept too: the FGSM evaluation attacks it, and
    # rebuilding it from the backbone alone would leave the final dense layer
    # randomly initialised, which makes the attack meaningless.
    cls_out = args.out.replace(".h5", "_cls.h5")
    model.save_weights(cls_out)

    json.dump({"finetune": args.finetune, "trainable_params": trainable,
               "backbone": args.backbone, "seed": args.seed,
               "classifier_weights": os.path.basename(cls_out)},
              open(args.out + ".json", "w"), indent=2)
    print("saved", args.out, "and", cls_out)

    for f in (args.out, cls_out, args.out + ".json"):
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            raise RuntimeError("checkpoint not written: %s" % f)
    return 0


def _exit_without_teardown(code):
    """Leave without running the interpreter's shutdown.

    model.fit over a Python generator leaves TensorFlow with worker threads
    that are still joinable when the C++ runtime tears down, which aborts with
    'terminate called without an active exception' and exit 134. That happens
    after the weights are on disk, so the run has in fact succeeded - but Slurm
    sees a non-zero exit and every job chained with afterok is skipped.
    Flushing and leaving directly avoids the broken shutdown path.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    _exit_without_teardown(main() or 0)

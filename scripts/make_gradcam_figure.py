"""Redraw the Grad-CAM panel from the MobileNetV2 the paper actually reports.

The figure carried through from the first version of this work came out of
legacy/gradcam_definitive.py, which ran on VGG19: its panels are labelled
"Block 1 Conv 2" to "Block 5 Conv 4", names that exist in VGG19 and nowhere in
MobileNetV2. The caption described five inverted-residual stages while the
picture showed something else. It came to light when a reviewer asked for the
visuals to be checked.

This regenerates the maps from the fine-tuned MobileNetV2 checkpoint, labels
the stages with the layer names the code actually reads, and overlays them on
the unmodified frame with a colormap where hot means high.

    python scripts/make_gradcam_figure.py --root $DATA_ROOT/hockey \\
        --backbone-weights ckpt/mnv2_ft5_cls.h5 --out figures/gradcam
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset to pick a clip from")
    ap.add_argument("--backbone-weights", required=True,
                    help="the frame classifier saved by finetune_backbone.py "
                         "(*_cls.h5): Grad-CAM needs a class output")
    ap.add_argument("--out", default="figures/gradcam")
    ap.add_argument("--clip", default=None,
                    help="specific video; otherwise the first violent clip")
    ap.add_argument("--frame", type=int, default=None,
                    help="frame index; otherwise the most confident frame")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--cols", type=int, default=3,
                    help="panels per row. 6 puts all of them on one line, "
                         "which is what fits across a two-column page")
    ap.add_argument("--width", type=float, default=6.6,
                    help="figure width in inches")
    args = ap.parse_args()

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data.datasets import index_flat, load_clip
    from src.models.build import build_backbone, build_classifier_head
    from src.xai.gradcam import MOBILENETV2_BLOCK_LAYERS, make_gradcam

    if args.clip:
        path, label = args.clip, 1
    else:
        items = index_flat(args.root)
        viol = [p for p, l in items if l == 1]
        if not viol:
            raise SystemExit("no violent clips under %s" % args.root)
        path, label = sorted(viol)[0], 1
    print("clip:", path)

    clip = load_clip(path, n_frames=args.frames, size=args.input_size)

    cnn = build_backbone("mobilenetv2", input_size=args.input_size,
                         weights="imagenet")
    model = build_classifier_head(cnn)
    model.load_weights(args.backbone_weights)
    print("loaded", args.backbone_weights)

    probs = model.predict(clip, batch_size=args.frames, verbose=0)[:, 1]
    idx = args.frame if args.frame is not None else int(np.argmax(probs))
    img = clip[idx]
    print("frame %d of %d, p(violent) = %.3f" % (idx, args.frames, probs[idx]))

    maps = {}
    for name in MOBILENETV2_BLOCK_LAYERS:
        try:
            maps[name] = make_gradcam(model, name, img, class_index=1)
        except ValueError as e:
            print("skipping %s (%s)" % (name, e))
    if not maps:
        raise SystemExit("no Grad-CAM layer could be read from this model")

    size = args.input_size
    big = {k: cv2.resize(v, (size, size)) for k, v in maps.items()}
    mean_map = np.mean(list(big.values()), axis=0)
    mean_map = mean_map / max(mean_map.max(), 1e-9)
    panels = list(big.items()) + [("mean of the five", mean_map)]

    n = len(panels)
    cols = max(1, min(args.cols, n))
    rowsn = int(np.ceil(n / cols))
    per = args.width / cols
    plt.rcParams.update({"font.size": 7.5 if cols <= 3 else 6.0,
                         "figure.dpi": 220})
    fig, axes = plt.subplots(rowsn, cols,
                             figsize=(args.width, (per + 0.22) * rowsn))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, cam) in zip(axes, panels):
        ax.imshow(img)
        # alpha follows the map, so where the network is not looking the frame
        # stays legible instead of being washed out by a flat overlay
        ax.imshow(cam, cmap="inferno", vmin=0.0, vmax=1.0,
                  alpha=0.25 + 0.5 * cam)
        ax.set_title(name, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout(h_pad=1.8 if rowsn > 1 else 0.4, w_pad=0.4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        p = "%s.%s" % (args.out, ext)
        fig.savefig(p, bbox_inches="tight")
        print("wrote", p)

    json.dump({"clip": path, "frame": int(idx),
               "p_violent": float(probs[idx]),
               "layers": list(maps), "weights": args.backbone_weights},
              open("%s.json" % args.out, "w"), indent=2)


if __name__ == "__main__":
    main()

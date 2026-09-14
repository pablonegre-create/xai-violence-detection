"""Download the ImageNet backbone weights into the Keras cache.

Compute nodes on the cluster have no outbound network, so the first call to
`keras.applications.MobileNetV2(weights="imagenet")` inside a job fails with a
URL error. Run this once on the login node: it populates ~/.keras/models,
which lives in the home directory and is therefore visible from the compute
nodes.

    python scripts/prefetch_weights.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# (builder name, input sizes actually used anywhere in the pipeline)
NEEDED = {
    "mobilenetv2": [128],
    "mobilenetv3small": [128],
    "efficientnetb0": [128],
    "resnet50": [128],
    "vgg19": [128],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["mobilenetv2"],
                    help="default is just the one the pipeline trains; pass "
                         "more if you plan to run the backbone ablation")
    ap.add_argument("--sizes", nargs="+", type=int, default=None)
    args = ap.parse_args()

    from src.models.build import build_backbone

    cache = os.path.join(os.path.expanduser("~"), ".keras", "models")
    print("cache directory:", cache)

    for name in args.backbones:
        sizes = args.sizes or NEEDED.get(name, [128])
        for s in sizes:
            print("fetching %s @ %dx%d ..." % (name, s, s), end=" ", flush=True)
            try:
                build_backbone(name, input_size=s, weights="imagenet")
                print("ok")
            except Exception as e:
                print("FAILED: %s" % e)

    print()
    if os.path.isdir(cache):
        for f in sorted(os.listdir(cache)):
            p = os.path.join(cache, f)
            if os.path.isfile(p):
                print("  %8.1f MB  %s" % (os.path.getsize(p) / 1e6, f))
    print("\nThese files must exist before submitting jobs 01, 02 and 07.")


if __name__ == "__main__":
    main()

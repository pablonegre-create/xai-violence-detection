"""Quantitative comparison of temporal explanation methods.

For every test clip we compute a per-frame attribution with each method and
score it by how well it ranks the evidence: deletion AUC (lower is better),
insertion AUC (higher is better), and the cost in model calls and seconds.
Rank agreement between methods is also reported, since block ablation and
TimeSHAP are estimating related but not identical quantities.

This is what answers "how was the explainability evaluated objectively" and
"how does your metric compare with TimeSHAP".

    python scripts/run_xai_eval.py --features features/rlvs \
        --head-weights ckpt/head_seed0.h5 --out results/xai_temporal.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--head-weights", required=True)
    ap.add_argument("--out", default="results/xai_temporal.json")
    ap.add_argument("--n-clips", type=int, default=100,
                    help="test clips to explain; -1 for all")
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--shapley-samples", type=int, default=40)
    ap.add_argument("--lstm", nargs="+", type=int, default=[64, 32])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from src.models.build import build_temporal_head
    from src.xai import temporal_baselines as tb
    from src.xai.frame_importance import importance_attribution, frame_importance
    from src.xai.metrics import (temporal_deletion_curve,
                                 temporal_insertion_curve, auc, rank_agreement)
    from scripts.train import load_split

    X, y, split, meta = load_split(args.features)
    te = np.where(split == "test")[0]
    rng = np.random.default_rng(args.seed)
    if args.n_clips > 0 and len(te) > args.n_clips:
        te = rng.choice(te, args.n_clips, replace=False)

    head = build_temporal_head(X.shape[1], X.shape[2],
                               lstm_units=tuple(args.lstm))
    head.load_weights(args.head_weights)

    methods = {
        "block_ablation": lambda f: importance_attribution(
            f, head, block=args.block),
        "occlusion": lambda f: tb.temporal_occlusion(f, head),
        "gradient_input": lambda f: tb.temporal_gradient_input(f, head),
        "integrated_gradients": lambda f: tb.temporal_integrated_gradients(
            f, head),
        "shapley_sampling": lambda f: tb.temporal_shapley_sampling(
            f, head, n_samples=args.shapley_samples, group=args.block),
        "timeshap": lambda f: tb.timeshap_event_attribution(f, head),
    }

    per_method = {m: {"del": [], "ins": [], "time": [], "attr": []}
                  for m in methods}

    for j, i in enumerate(te):
        feats = X[i]
        for name, fn in methods.items():
            t0 = time.perf_counter()
            try:
                attr = fn(feats)
            except Exception as e:
                print("  %s failed on clip %d: %s" % (name, i, e))
                continue
            dt = time.perf_counter() - t0

            d = temporal_deletion_curve(feats, head, attr)
            s = temporal_insertion_curve(feats, head, attr)

            per_method[name]["del"].append(auc(d))
            per_method[name]["ins"].append(auc(s))
            per_method[name]["time"].append(dt)
            per_method[name]["attr"].append(np.asarray(attr))

        if (j + 1) % 10 == 0:
            print("  %d/%d clips" % (j + 1, len(te)), flush=True)

    # random ordering as the floor for deletion/insertion
    rnd_del, rnd_ins = [], []
    for i in te:
        attr = rng.normal(size=X.shape[1])
        rnd_del.append(auc(temporal_deletion_curve(X[i], head, attr)))
        rnd_ins.append(auc(temporal_insertion_curve(X[i], head, attr)))

    summary = {"n_clips": int(len(te)), "block": args.block, "methods": {}}
    for name, d in per_method.items():
        if not d["del"]:
            continue
        n_calls = (X.shape[1] - args.block + 1 + 1 if name == "block_ablation"
                   else None)
        summary["methods"][name] = {
            "deletion_auc_mean": float(np.mean(d["del"])),
            "deletion_auc_std": float(np.std(d["del"])),
            "insertion_auc_mean": float(np.mean(d["ins"])),
            "insertion_auc_std": float(np.std(d["ins"])),
            "gap": float(np.mean(d["ins"]) - np.mean(d["del"])),
            "time_s_mean": float(np.mean(d["time"])),
            "time_s_std": float(np.std(d["time"])),
            "model_calls": n_calls,
        }
    summary["methods"]["random"] = {
        "deletion_auc_mean": float(np.mean(rnd_del)),
        "insertion_auc_mean": float(np.mean(rnd_ins)),
        "gap": float(np.mean(rnd_ins) - np.mean(rnd_del)),
    }

    # how much do the methods agree with the proposed one
    ref = "block_ablation"
    summary["agreement_with_%s" % ref] = {}
    if per_method[ref]["attr"]:
        for name in methods:
            if name == ref or not per_method[name]["attr"]:
                continue
            rhos, taus, jac = [], [], []
            for a, b in zip(per_method[ref]["attr"], per_method[name]["attr"]):
                r = rank_agreement(a, b, k=5)
                rhos.append(r["spearman"]); taus.append(r["kendall"])
                jac.append(r["top5_jaccard"])
            summary["agreement_with_%s" % ref][name] = {
                "spearman_mean": float(np.nanmean(rhos)),
                "kendall_mean": float(np.nanmean(taus)),
                "top5_jaccard_mean": float(np.mean(jac)),
            }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary["methods"], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

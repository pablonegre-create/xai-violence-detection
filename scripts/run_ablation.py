"""Component ablation.

Each row removes or swaps one piece of the pipeline and retrains the head over
the same seeds, so the differences can be tested with a paired test rather
than eyeballed. The backbone variants need their own cached features, which
scripts/extract_features.py produces; the head variants all reuse one cache.

    python scripts/run_ablation.py --features features/rlvs \
        --out results/ablation_rlvs.json
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# name -> extra args for scripts/train.py
HEAD_VARIANTS = {
    "full (2 Bi-LSTM + 2 FC)":      [],
    "1 Bi-LSTM":                    ["--lstm", "64"],
    "3 Bi-LSTM":                    ["--lstm", "64", "32", "16"],
    "Bi-GRU instead of Bi-LSTM":    ["--head", "bigru"],
    "no recurrence (avg pooling)":  ["--head", "avgpool"],
    "with batch norm":              ["--batch-norm"],
    "1 FC layer":                   ["--dense", "64"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="results/ablation.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tmp", default="results/_ablation_tmp")
    args = ap.parse_args()

    from src.eval.stats import paired_ttest_seeds, holm_bonferroni

    os.makedirs(args.tmp, exist_ok=True)
    rows = {}

    for name, extra in HEAD_VARIANTS.items():
        tag = name.replace(" ", "_").replace("(", "").replace(")", "")
        out = os.path.join(args.tmp, tag + ".json")
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "train.py"),
               "--features", args.features, "--seeds", str(args.seeds),
               "--out", out] + extra
        print("\n>>", name)
        subprocess.run(cmd, check=True)
        rows[name] = json.load(open(out))

    ref = "full (2 Bi-LSTM + 2 FC)"
    ref_accs = [r["accuracy"] for r in rows[ref]["runs"]]

    table, pvals, keys = [], [], []
    for name, res in rows.items():
        accs = [r["accuracy"] for r in res["runs"]]
        entry = {
            "variant": name,
            "accuracy_mean": res["accuracy"]["mean"],
            "accuracy_std": res["accuracy"]["std"],
            "f1_mean": res["f1"]["mean"],
            "delta_vs_full": res["accuracy"]["mean"] - rows[ref]["accuracy"]["mean"],
        }
        if name != ref:
            t = paired_ttest_seeds(ref_accs, accs)
            entry.update({"t": t["t"], "p_value": t["p_value"],
                          "cohens_d": t["cohens_d"]})
            pvals.append(t["p_value"]); keys.append(name)
        table.append(entry)

    if pvals:
        adj, sig = holm_bonferroni(pvals)
        lut = dict(zip(keys, zip(adj, sig)))
        for e in table:
            if e["variant"] in lut:
                e["p_holm"], e["significant"] = float(lut[e["variant"]][0]), bool(lut[e["variant"]][1])

    json.dump({"reference": ref, "seeds": args.seeds, "table": table},
              open(args.out, "w"), indent=2)

    print("\n%-32s %8s %8s %9s %9s" % ("variant", "acc", "sd", "delta", "p(Holm)"))
    for e in table:
        print("%-32s %8.4f %8.4f %9.4f %9s" % (
            e["variant"], e["accuracy_mean"], e["accuracy_std"],
            e["delta_vs_full"],
            ("%.4f" % e["p_holm"]) if "p_holm" in e else "-"))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

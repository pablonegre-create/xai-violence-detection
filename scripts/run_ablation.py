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
    ap.add_argument("--equivalence-margin", nargs="+", type=float,
                    default=[0.01, 0.02],
                    help="TOST margins in accuracy, fixed before looking. One "
                         "point is the tightest claim worth making; two points "
                         "is the difference below which we would not change "
                         "the architecture recommendation")
    args = ap.parse_args()

    from src.eval.stats import (paired_ttest_seeds, holm_bonferroni,
                                tost_equivalence, paired_diff_ci)

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
            # a large p-value is not evidence of equivalence, so report the
            # interval the data allow and a TOST against a stated margin
            entry["ci_vs_full"] = paired_diff_ci(accs, ref_accs)
            entry["tost"] = {("%.3f" % m): tost_equivalence(accs, ref_accs,
                                                            margin=m)
                             for m in args.equivalence_margin}
            pvals.append(t["p_value"]); keys.append(name)
        table.append(entry)

    if pvals:
        adj, sig = holm_bonferroni(pvals)
        lut = dict(zip(keys, zip(adj, sig)))
        for e in table:
            if e["variant"] in lut:
                e["p_holm"], e["significant"] = float(lut[e["variant"]][0]), bool(lut[e["variant"]][1])

    json.dump({"reference": ref, "seeds": args.seeds,
               "equivalence_margin": args.equivalence_margin, "table": table},
              open(args.out, "w"), indent=2)

    margins = ["%.3f" % m for m in args.equivalence_margin]
    print("\n%-32s %8s %8s %9s %9s %19s %s"
          % ("variant", "acc", "sd", "delta", "p(Holm)", "95% CI on diff",
             "  ".join("TOST+-%s" % m for m in margins)))
    for e in table:
        ci, tost = e.get("ci_vs_full"), e.get("tost")
        eq = "  ".join(
            ("%-9s" % ("yes" if tost[m]["equivalent"] else "no"))
            for m in margins) if tost else "-"
        print("%-32s %8.4f %8.4f %9.4f %9s %19s %s" % (
            e["variant"], e["accuracy_mean"], e["accuracy_std"],
            e["delta_vs_full"],
            ("%.4f" % e["p_holm"]) if "p_holm" in e else "-",
            "-" if not ci else "[%+.4f, %+.4f]" % (ci["lo"], ci["hi"]), eq))
    print("\nTOST at alpha=0.05 against margins %s accuracy, %d seeds"
          % (", ".join("+-" + m for m in margins), args.seeds))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

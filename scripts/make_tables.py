"""Emit the manuscript tables straight from the result files.

Typing numbers out of JSON into LaTeX by hand is how a paper ends up with a
figure that does not match its own data. Every table in the manuscript and in
Online Resource 1 is generated here instead.

    python scripts/make_tables.py --results results --out tables.tex
"""

import argparse
import json
import math
import os
import sys

NAMES = {
    "rlvs": "RLVS", "rwf2000": "RWF-2000", "hockey": "Hockey Fights",
    "movies": "Movies Fight", "violentflows": "Violent Flows",
}
ORDER = ["rlvs", "rwf2000", "hockey", "movies", "violentflows"]
SHORT = {"rlvs": "RLVS", "rwf2000": "RWF", "hockey": "Hock.",
         "movies": "Mov.", "violentflows": "VF"}


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main_table(R):
    rows = []
    for d in ORDER:
        j = load(R, "main_%s.json" % d)
        if not j:
            continue
        a, f, u = j["accuracy"], j["f1"], j["auc"]
        lo, hi = a.get("wilson95", (float("nan"),) * 2)
        rows.append("%s & %.1f $\\pm$ %.1f & %.3f & %.3f & [%.1f, %.1f] \\\\"
                    % (NAMES[d], 100 * a["mean"], 100 * a["std"],
                       f["mean"], u["mean"], 100 * lo, 100 * hi))
    return "\n".join(rows)


def cross_table(R):
    j = load(R, "cross_dataset.json")
    if not j:
        return ""
    ds = j["datasets"]
    out = []
    for s in ds:
        cells = []
        for t in ds:
            m = j["matrix"][s][t]
            v = "%.1f" % (100 * m["accuracy_mean"])
            cells.append("\\textbf{%s}" % v if m["in_domain"] else v)
        out.append("%s & %s \\\\" % (SHORT.get(s, s), " & ".join(cells)))
    return "\n".join(out)


def ablation_table(R):
    j = load(R, "ablation_rlvs.json")
    if not j:
        return ""
    out = []
    for e in j["table"]:
        p = "%.2f" % e["p_holm"] if "p_holm" in e else "---"
        dl = "---" if e["variant"] == j["reference"] else \
             "%+.2f" % (100 * e["delta_vs_full"])
        out.append("%s & %.2f $\\pm$ %.2f & %s & %s \\\\"
                   % (e["variant"], 100 * e["accuracy_mean"],
                      100 * e["accuracy_std"], dl, p))
    return "\n".join(out)


def probe_table(R):
    """Order probe, one file per dataset.

    Runs made before the verdict was tied to sampling noise have no
    noise_floor_95 field, so recompute it here from the split the file itself
    records. It is the same expression run_temporal_probe.py uses and it needs
    nothing but n_test and the untransformed accuracy.
    """
    out = []
    for d in ORDER:
        j = load(R, "temporal_probe%s.json" % ("" if d == "rlvs" else "_" + d))
        if not j:
            continue
        t = j["transforms"]
        n, p = j["n_test"], t["identity"]["accuracy"]
        noise = j.get("noise_floor_95", 1.96 * math.sqrt(p * (1 - p) / n))
        # two decimals on the deltas: on the 400-clip splits one clip is 0.25
        # points, and rounding that to 0.2/0.3 hides which rows are exact zeros
        out.append("%s & %d & %.1f & %+.2f & %+.2f & %+.2f & $\\pm$%.2f \\\\"
                   % (NAMES[d], n, 100 * p,
                      100 * t["shuffle"]["delta"], 100 * t["reverse"]["delta"],
                      100 * t["sort_by_norm"]["delta"], 100 * noise))
    return "\n".join(out)


def efficiency_table(R, which="complexity_cn001_cpu.json"):
    j = load(R, which)
    if not j:
        return ""
    label = {"mobilenetv2": "MobileNetV2", "mobilenetv3small": "MobileNetV3-S",
             "efficientnetb0": "EfficientNet-B0", "resnet50": "ResNet-50",
             "vgg19": "VGG-19"}
    out = []
    for r in j["rows"]:
        out.append("%s & %.2f & %.1f & %.1f & %.1f & %.0f \\\\"
                   % (label.get(r["backbone"], r["backbone"]),
                      r["params_total_M"], r["size_fp32_MB"],
                      r["flops_clip_G"], r["latency_frame_ms"],
                      r["latency_clip_batch_ms"]))
    return "\n".join(out)


def xai_table(R):
    j = load(R, "xai_temporal_rlvs.json")
    if not j:
        return ""
    label = {"block_ablation": "Block ablation (ours)",
             "occlusion": "Temporal occlusion",
             "gradient_input": "Gradient $\\times$ input",
             "integrated_gradients": "Integrated gradients",
             "shapley_sampling": "Sampled Shapley",
             "timeshap": "TimeSHAP", "random": "Random ordering"}
    agree = j.get("agreement_with_block_ablation", {})
    out = []
    for k, v in j["methods"].items():
        rho = agree.get(k, {}).get("spearman_mean")
        out.append("%s & %.3f & %.3f & %.3f & %s & %.2f \\\\"
                   % (label.get(k, k), v["deletion_auc_mean"],
                      v["insertion_auc_mean"], v["gap"],
                      "---" if rho is None else "%.2f" % rho,
                      v.get("time_s_mean", 0.0)))
    return "\n".join(out)


def synthetic_table(R):
    j = load(R, "synthetic_xai_benchmark.json")
    if not j:
        return ""
    label = {"block_ablation": "Block ablation (ours)",
             "occlusion": "Temporal occlusion",
             "gradient_input": "Gradient $\\times$ input",
             "integrated_gradients": "Integrated gradients",
             "shapley_sampling": "Sampled Shapley",
             "random": "Random ordering"}
    out = []
    for k, v in j["aggregate"].items():
        out.append("%s & %.3f $\\pm$ %.3f & %.3f & %.3f & %.2f \\\\"
                   % (label.get(k, k), v["precision_at_k_mean"],
                      v["precision_at_k_std"], v["pointing_mean"],
                      v["mass_on_event_mean"], v["time_s_mean"]))
    return "\n".join(out)


def robustness_table(R, severities=(1, 3, 5)):
    j = load(R, "robustness_rlvs.json")
    if not j:
        return ""
    label = {"gaussian_noise": "Gaussian noise",
             "low_illumination": "Low illumination", "motion_blur": "Motion blur",
             "camera_shake": "Camera shake", "occlusion_static": "Occlusion (static)",
             "occlusion_moving": "Occlusion (moving)", "compression": "Compression",
             "frame_drop": "Frame dropping", "downscale": "Downscaling",
             "rain": "Rain"}
    seen, out = [], []
    for r in j["rows"]:
        if r["corruption"] not in seen:
            seen.append(r["corruption"])
    for c in seen:
        cells = []
        for s in severities:
            m = [r for r in j["rows"]
                 if r["corruption"] == c and r["severity"] == s]
            cells.append("%.1f" % (100 * m[0]["accuracy"]) if m else "---")
        out.append("%s & %s \\\\" % (label.get(c, c), " & ".join(cells)))
    if "adversarial_fgsm" in j:
        adv = " & ".join("%.1f" % (100 * a["accuracy"])
                         for a in j["adversarial_fgsm"][:3])
        out.append("\\midrule\nFGSM $\\epsilon$=1,2,4/255 & %s \\\\" % adv)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    R = args.results
    blocks = [
        ("Table 1 -- detection performance", main_table(R)),
        ("Table 2 -- cross-dataset transfer", cross_table(R)),
        ("Table 3 -- component ablation", ablation_table(R)),
        ("Table 4 -- order probe", probe_table(R)),
        ("Table 5 -- efficiency (CPU)", efficiency_table(R)),
        ("Table 5b -- efficiency (H100, ESM)",
         efficiency_table(R, "complexity_h100.json")),
        ("Table 6 -- temporal attribution on RLVS", xai_table(R)),
        ("Table 7 -- synthetic attribution benchmark", synthetic_table(R)),
        ("Table 8 -- robustness", robustness_table(R)),
    ]

    text = []
    for title, body in blocks:
        text.append("%% ===== %s =====" % title)
        text.append(body if body else "%% (missing result file)")
        text.append("")
    text = "\n".join(text)

    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
        print("wrote", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()

"""Emit the manuscript tables straight from the result files.

Typing numbers out of JSON into LaTeX by hand is how a paper ends up with a
figure that does not match its own data. Every table of results in the
manuscript and in Online Resource 1 is generated here instead, row for row in
the layout the paper prints, so a reader can regenerate a table and compare it
with the published one line by line.

    python scripts/make_tables.py --results results --out tables.tex

Each block is headed by the LaTeX label of the table it fills.
"""

import argparse
import json
import math
import os
from decimal import Decimal, ROUND_HALF_UP

NAMES = {
    "rlvs": "RLVS", "rwf2000": "RWF-2000", "hockey": "Hockey Fights",
    "movies": "Movies Fight", "violentflows": "Violent Flows",
    "industrial": "Industrial Surv.",
}
# the main-text tables are a single column wide
MAIN_NAMES = {
    "rlvs": "RLVS", "rwf2000": "RWF-2000", "hockey": "Hockey",
    "movies": "Movies", "violentflows": "Violent Flows",
    "industrial": "Industrial",
}
ORDER = ["rlvs", "rwf2000", "hockey", "movies", "violentflows", "industrial"]
TRANSFER_NAMES = {"rlvs": "RLVS", "rwf2000": "RWF-2000", "hockey": "Hockey",
                  "movies": "Movies", "violentflows": "Violent Flows"}

METHOD = {"block_ablation": "Block ablation (proposed)",
          "occlusion": "Temporal occlusion",
          "gradient_input": "Gradient $\\times$ input",
          "integrated_gradients": "Integrated gradients",
          "shapley_sampling": "Sampled Shapley",
          "timeshap": "TimeSHAP", "random": "Random ordering"}
METHOD_ORDER = ["block_ablation", "occlusion", "gradient_input",
                "integrated_gradients", "shapley_sampling", "timeshap",
                "random"]

ABLATION_NAMES = {"full (2 Bi-LSTM + 2 FC)": "Full",
                  "1 Bi-LSTM": "1 Bi-LSTM", "3 Bi-LSTM": "3 Bi-LSTM",
                  "Bi-GRU instead of Bi-LSTM": "Bi-GRU",
                  "no recurrence (avg pooling)": "Mean pooling",
                  "with batch norm": "+ batch norm",
                  "1 FC layer": "1 FC layer"}

ARMS = [("appearance (rgb)", "frames"),
        ("motion (frame difference)", "differences"),
        ("both (concatenated)", "both")]

# the two corruptions that leave every frame intact first, as in the paper
ROBUSTNESS_ORDER = ["camera_shake", "frame_drop", "compression",
                    "occlusion_moving", "occlusion_static", "downscale",
                    "motion_blur", "gaussian_noise", "rain",
                    "low_illumination"]
ROBUSTNESS_NAMES = {"gaussian_noise": "Gaussian noise",
                    "low_illumination": "Low illumination",
                    "motion_blur": "Motion blur",
                    "camera_shake": "Camera shake",
                    "occlusion_static": "Occlusion (static)",
                    "occlusion_moving": "Occlusion (moving)",
                    "compression": "Compression",
                    "frame_drop": "Frame dropping",
                    "downscale": "Downscaling", "rain": "Rain"}

BACKBONES = {"mobilenetv2": "MobileNetV2", "mobilenetv3small": "MobileNetV3-S",
             "efficientnetb0": "EfficientNet-B0", "resnet50": "ResNet-50",
             "vgg19": "VGG-19"}


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fx(x, nd=1, sign=False, lead=True):
    """Format a number the way the paper prints it.

    Rounds half up. Python's own rounding works on the binary value, so 96.125
    can come out as 96.12 in the same table where 96.275 comes out as 96.28;
    the results are short decimals (accuracies over a few hundred clips) and
    should round as such. Negative numbers get a typeset minus, `sign` adds a
    plus to positive ones, and `lead=False` drops the leading zero, which the
    single-column tables of the main text need to fit.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    d = Decimal(repr(round(x, 9))).quantize(Decimal(1).scaleb(-nd),
                                            rounding=ROUND_HALF_UP)
    if d == 0:
        d = abs(d)
    t = format(d, "f")
    neg = t.startswith("-")
    if neg:
        t = t[1:]
    if not lead and t.startswith("0."):
        t = t[1:]
    if neg:
        return "$-$" + t
    return ("$+$" + t) if (sign and d != 0) else t


def pct(x, nd=1, sign=False):
    return fx(None if x is None else 100 * x, nd, sign)


def probe_file(d):
    return "temporal_probe%s.json" % ("" if d == "rlvs" else "_" + d)


# ------------------------------------------------------------- main text

def main_table(R):
    """Table 1: precision, recall, F1 and AUC without the leading zero."""
    def short(v):
        s = fx(v, 3, lead=False)
        return "1.00" if s == "1.000" else s
    rows = []
    for d in ORDER:
        j = load(R, "main_%s.json" % d)
        if not j:
            continue
        a = j["accuracy"]
        rows.append("%s & %s $\\pm$ %s & %s & %s & %s & %s \\\\"
                    % (MAIN_NAMES[d], pct(a["mean"]), pct(a["std"]),
                       short(j["precision"]["mean"]),
                       short(j["recall"]["mean"]), short(j["f1"]["mean"]),
                       short(j["auc"]["mean"])))
    return "\n".join(rows)


def wilson_line(R):
    """The intervals Table 1 gives in its caption."""
    out = []
    for d in ORDER:
        j = load(R, "main_%s.json" % d)
        if j:
            w = j["accuracy"]["wilson95"]
            out.append("%s [%s, %s]" % (MAIN_NAMES[d], pct(w[0]), pct(w[1])))
    return "% " + "; ".join(out)


def ablation_table(R):
    """Table 2: accuracy over the seeds and the interval on the paired
    difference against the full model, in points."""
    j = load(R, "ablation_rlvs.json")
    if not j:
        return ""
    out = []
    for e in j["table"]:
        ci = e.get("ci_vs_full")
        ci_s = ("[%s, %s]" % (fx(100 * ci["lo"], 2, True, lead=False),
                              fx(100 * ci["hi"], 2, True, lead=False))
                if ci and e["variant"] != j["reference"] else "---")
        out.append("%s & %s $\\pm$ %s & %s \\\\"
                   % (ABLATION_NAMES.get(e["variant"], e["variant"]),
                      pct(e["accuracy_mean"], 2), pct(e["accuracy_std"], 2),
                      ci_s))
    return "\n".join(out)


def ablation_tost(R):
    """The equivalence tests the text of Section 5.2 quotes."""
    j = load(R, "ablation_rlvs.json")
    if not j:
        return ""
    out = []
    for e in j["table"]:
        for m, t in sorted((e.get("tost") or {}).items()):
            out.append("%% %s, margin %s: p = %.4f, equivalent: %s"
                       % (e["variant"], m, t["p_value"], t["equivalent"]))
    return "\n".join(out)


def splice_table(R):
    """Table 3: attribution on real footage with a spliced event, the four
    columns that fit a single column of the page. Time keeps three decimals:
    rounded to two, block ablation reads 0.08 and the ratio to Shapley comes
    out as 173 instead of the 163 the text quotes."""
    j = load(R, "semisynthetic_xai.json")
    if not j:
        return ""
    short = {"block_ablation": "Block ablation", "occlusion": "Occlusion",
             "gradient_input": "Gradient $\\times$ input",
             "integrated_gradients": "Integrated gradients",
             "shapley_sampling": "Sampled Shapley", "random": "Random"}
    out = []
    for k in METHOD_ORDER:
        v = j["aggregate"].get(k)
        if not v:
            continue
        out.append("%s & %s & %s & %s & %s \\\\"
                   % (short[k], fx(v["precision_at_k_mean"], 3),
                      fx(v["pointing_mean"], 3), fx(v["mass_on_event_mean"], 3),
                      "---" if k == "random" else fx(v["time_s_mean"], 3)))
    return "\n".join(out)


def motion_table(R, datasets=("rwf2000", "rlvs", "industrial")):
    """Table 4: the three inputs, the paired difference against the frames
    arm, and the verdict of the order probe on each arm."""
    out = []
    for d in datasets:
        j = load(R, "motion_control_%s.json" % d)
        if not j:
            continue
        out.append("\\multicolumn{4}{@{}l}{\\textit{%s}} \\\\" % NAMES[d])
        for key, lab in ARMS:
            a = j["arms"][key]
            vs = a.get("vs_appearance")
            dl = "---"
            if vs:
                ci = vs["ci"]
                dl = "%s [%s, %s]" % (pct(ci["mean_diff"], 1, True),
                                      pct(ci["lo"]), pct(ci["hi"]))
            sens = a.get("order_probe", {}).get("order_sensitive")
            out.append("\\quad %s & %s $\\pm$ %s & %s & %s \\\\"
                       % (lab, pct(a["accuracy"]["mean"]),
                          pct(a["accuracy"]["std"]), dl,
                          "---" if sens is None
                          else ("sensitive" if sens else "invariant")))
    return "\n".join(out)


# ------------------------------------------------------ Online Resource 1

def probe_table(R):
    """tab:probefull. Two decimals on the deltas: on the 400-clip splits one
    clip is 0.25 points, and rounding that to 0.2/0.3 hides which rows are
    exact zeros.

    Runs made before the verdict was tied to sampling noise have no
    noise_floor_95 field, so it is recomputed from the split the file records,
    with the same expression run_temporal_probe.py uses.
    """
    out = []
    for d in ORDER:
        j = load(R, probe_file(d))
        if not j:
            continue
        t = j["transforms"]
        n, p = j["n_test"], t["identity"]["accuracy"]
        noise = j.get("noise_floor_95", 1.96 * math.sqrt(p * (1 - p) / n))
        out.append("%s & %d & %s & %s & %s & %s & $\\pm$%s \\\\"
                   % (NAMES[d], n, pct(p), pct(t["shuffle"]["delta"], 2, True),
                      pct(t["reverse"]["delta"], 2, True),
                      pct(t["sort_by_norm"]["delta"], 2, True),
                      pct(noise, 2)))
    return "\n".join(out)


def probe_function_table(R):
    """tab:probefunc: what the shuffle does per clip, which accuracy alone
    cannot show."""
    out = []
    for d in ORDER:
        j = load(R, probe_file(d))
        if not j:
            continue
        sh = j["transforms"]["shuffle"]
        if "flip_rate" not in sh:
            continue
        out.append("%s & %d & %s $\\pm$ %s & %s & %s & %s & %s \\\\"
                   % (NAMES[d], j.get("n_perm", 1), pct(sh["delta"], 2, True),
                      pct(sh.get("accuracy_sd", 0.0), 2),
                      pct(sh["flip_rate"]), fx(sh["abs_dp_mean"], 4),
                      fx(sh["abs_dp_p95"], 4), fx(sh.get("prob_corr"), 3)))
    return "\n".join(out)


def kfold_table(R):
    """tab:kfold: 10-fold next to the holdout the paper reports."""
    out = []
    for d in ORDER:
        k = load(R, "kfold_%s.json" % d)
        m = load(R, "main_%s.json" % d)
        if not k or not m:
            continue
        a, h = k["accuracy"], m["accuracy"]
        out.append("%s & %s $\\pm$ %s & %s $\\pm$ %s & %s & %s \\\\"
                   % (NAMES[d], pct(h["mean"]), pct(h["std"]), pct(a["mean"]),
                      pct(a["std"]), pct(a["min"]), pct(a["max"])))
    return "\n".join(out)


def cross_table(R):
    """tab:transfer, with the two averages the text quotes."""
    j = load(R, "cross_dataset.json")
    if not j:
        return ""
    ds = j["datasets"]
    out = []
    for s in ds:
        cells = []
        for t in ds:
            v = pct(j["matrix"][s][t]["accuracy_mean"])
            cells.append("\\textbf{%s}" % v if s == t else v)
        out.append("%s & %s \\\\" % (TRANSFER_NAMES.get(s, s), " & ".join(cells)))
    ind = [j["matrix"][s][s]["accuracy_mean"] for s in ds]
    off = [j["matrix"][s][t]["accuracy_mean"] for s in ds for t in ds if s != t]
    out.append("%% in-domain mean %s, out-of-domain mean %s"
               % (pct(sum(ind) / len(ind)), pct(sum(off) / len(off))))
    return "\n".join(out)


def motion_full_table(R, datasets=("rwf2000", "rlvs", "industrial")):
    """tab:motionfull: adds AUC and the probe of every arm."""
    out = []
    for d in datasets:
        j = load(R, "motion_control_%s.json" % d)
        if not j:
            continue
        out.append("\\multicolumn{7}{@{}l}{\\textit{%s}} \\\\" % NAMES[d])
        for key, lab in ARMS:
            a = j["arms"][key]
            pr = a["order_probe"]
            vs = a.get("vs_appearance")
            dl = "---"
            if vs:
                ci = vs["ci"]
                dl = "%s [%s, %s]" % (pct(ci["mean_diff"], 1, True),
                                      pct(ci["lo"]), pct(ci["hi"]))
            out.append("\\quad %s & %s $\\pm$ %s & %s & %s & %s & $\\pm$%s & %s \\\\"
                       % (lab, pct(a["accuracy"]["mean"]),
                          pct(a["accuracy"]["std"]), fx(a["auc"]["mean"], 3),
                          dl, pct(pr["delta"], 1, True),
                          pct(pr["noise_floor_95"]), pct(pr["flip_rate"])))
    return "\n".join(out)


def robustness_table(R, severities=(1, 3, 5)):
    """tab:robustfull: accuracy, change and relative degradation (the share
    of the accuracy above chance that is lost) per severity."""
    j = load(R, "robustness_rlvs.json")
    if not j:
        return ""
    out = []
    for c in ROBUSTNESS_ORDER:
        cells = []
        for s in severities:
            m = [r for r in j["rows"]
                 if r["corruption"] == c and r["severity"] == s]
            if not m:
                cells += ["---"] * 3
                continue
            r = m[0]
            cells += [pct(r["accuracy"]), pct(-r["drop"], 1, True),
                      fx(r["relative_error"], 2)]
        out.append("%s & %s \\\\" % (ROBUSTNESS_NAMES.get(c, c),
                                     " & ".join(cells)))
    return "\n".join(out)


def adversarial_table(R):
    """tab:adv."""
    j = load(R, "robustness_rlvs.json")
    if not j or "adversarial_fgsm" not in j:
        return ""
    return "Accuracy & %s \\\\" % " & ".join(
        pct(a["accuracy"], 2) for a in j["adversarial_fgsm"])


def xai_table(R):
    """tab:rlvsxai: deletion and insertion on RLVS, and agreement with block
    ablation."""
    j = load(R, "xai_temporal_rlvs.json")
    if not j:
        return ""
    agree = j.get("agreement_with_block_ablation", {})
    out = []
    for k in METHOD_ORDER:
        v = j["methods"].get(k)
        if not v:
            continue
        rho = agree.get(k, {}).get("spearman_mean")
        out.append("%s & %s & %s & %s & %s & %s \\\\"
                   % (METHOD[k], fx(v["deletion_auc_mean"], 3),
                      fx(v["insertion_auc_mean"], 3), fx(v["gap"], 3),
                      fx(rho, 2),
                      "---" if k == "random" else fx(v.get("time_s_mean"), 2)))
    return "\n".join(out)


def synthetic_table(R):
    """tab:synthetic: planted event in Gaussian descriptors, five seeds."""
    j = load(R, "synthetic_xai_benchmark.json")
    if not j:
        return ""
    out = []
    for k in METHOD_ORDER:
        v = j["aggregate"].get(k)
        if not v:
            continue
        out.append("%s & %s $\\pm$ %s & %s & %s & %s & %s & %s \\\\"
                   % (METHOD[k], fx(v["precision_at_k_mean"], 3),
                      fx(v["precision_at_k_std"], 3), fx(v["pointing_mean"], 3),
                      fx(v["mass_on_event_mean"], 3),
                      fx(v["spearman_gt_mean"], 3),
                      fx(v["deletion_auc_mean"], 3),
                      "---" if k == "random" else fx(v["time_s_mean"], 3)))
    return "\n".join(out)


def semisynthetic_table(R):
    """tab:semisynth: planted event in real footage, one seed, spread over
    the explained clips."""
    j = load(R, "semisynthetic_xai.json")
    if not j:
        return ""
    out = []
    for k in METHOD_ORDER:
        v = j["aggregate"].get(k)
        if not v:
            continue
        out.append("%s & %s $\\pm$ %s & %s & %s & %s & %s & %s & %s \\\\"
                   % (METHOD[k], fx(v["precision_at_k_mean"], 3),
                      fx(v["precision_at_k_std"], 3), fx(v["pointing_mean"], 3),
                      fx(v["mass_on_event_mean"], 3),
                      fx(v["spearman_gt_mean"], 3),
                      fx(v["deletion_auc_mean"], 3),
                      fx(v["insertion_auc_mean"], 3),
                      "---" if k == "random" else fx(v["time_s_mean"], 3)))
    sh = j["aggregate"]["shapley_sampling"]["time_s_mean"]
    bl = j["aggregate"]["block_ablation"]["time_s_mean"]
    out.append("%% head accuracy %s, sampled Shapley costs %dx block ablation"
               % (pct(j["head_accuracy"]), int(sh / bl)))
    return "\n".join(out)


def efficiency_table(R):
    """tab:h100: the same measurement on the cluster's CPU and on an H100."""
    cpu = load(R, "complexity_cn001_cpu.json")
    gpu = load(R, "complexity_h100.json")
    if not cpu or not gpu:
        return ""
    g = {r["backbone"]: r for r in gpu["rows"]}
    out = []
    for r in cpu["rows"]:
        h = g.get(r["backbone"], {})
        out.append("%s & %s & %s & %s & %s & %s & %s & %s \\\\"
                   % (BACKBONES.get(r["backbone"], r["backbone"]),
                      fx(r["params_total_M"], 2), fx(r["size_fp32_MB"], 1),
                      fx(r["flops_clip_G"], 1), fx(r["latency_frame_ms"], 2),
                      fx(r["latency_clip_batch_ms"], 1),
                      fx(h.get("latency_frame_ms"), 2),
                      fx(h.get("latency_clip_batch_ms"), 1)))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    R = args.results
    blocks = [
        ("main text, tab:main", main_table(R)),
        ("main text, tab:main caption (Wilson intervals)", wilson_line(R)),
        ("main text, tab:ablation", ablation_table(R)),
        ("main text, Section 5.2 (equivalence tests)", ablation_tost(R)),
        ("main text, tab:splice", splice_table(R)),
        ("main text, tab:motion", motion_table(R)),
        ("ESM, tab:probefull", probe_table(R)),
        ("ESM, tab:probefunc", probe_function_table(R)),
        ("ESM, tab:kfold", kfold_table(R)),
        ("ESM, tab:transfer", cross_table(R)),
        ("ESM, tab:motionfull", motion_full_table(R)),
        ("ESM, tab:robustfull", robustness_table(R)),
        ("ESM, tab:adv", adversarial_table(R)),
        ("ESM, tab:rlvsxai", xai_table(R)),
        ("ESM, tab:synthetic", synthetic_table(R)),
        ("ESM, tab:semisynth", semisynthetic_table(R)),
        ("ESM, tab:h100", efficiency_table(R)),
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

"""Draw the two result figures from the JSON in results/.

Same reasoning as make_tables.py: a plot redrawn by hand drifts away from the
numbers it is supposed to show. Both panels come straight out of the files.

    python scripts/make_figures.py --results results --out fig

Needs matplotlib and nothing else; no GPU, no dataset, no checkpoints.
"""

import argparse
import json
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["rlvs", "rwf2000", "hockey", "movies", "violentflows",
         "industrial"]
SHORT = {"rlvs": "RLVS", "rwf2000": "RWF-2000", "hockey": "Hockey",
         "movies": "Movies", "violentflows": "Violent Flows",
         "industrial": "Industrial Surv."}

# the three transforms, in the order the probe applies them
TRANSFORMS = [("shuffle", "o", "shuffle"),
              ("reverse", "^", "reverse"),
              ("sort_by_norm", "D", "sort by norm")]

# corruptions that destroy temporal structure but leave each frame intact,
# against ones that damage appearance. The split is the point of the panel.
TEMPORAL = [("camera_shake", "-o", "camera shake"),
            ("frame_drop", "--s", "frame dropping")]
APPEARANCE = [("gaussian_noise", ":d", "Gaussian noise"),
              ("low_illumination", ":v", "low illumination"),
              ("rain", ":s", "rain")]


def load(d, name, required=True):
    """Read a result file. Panels that iterate over datasets pass
    required=False, so the figure draws whatever has been run rather than
    refusing to draw anything."""
    p = os.path.join(d, name)
    if not os.path.exists(p):
        if required:
            raise SystemExit("missing result file: %s" % p)
        return None
    return json.load(open(p))


def probe_panel(ax, R):
    present = [d for d in ORDER
               if load(R, "temporal_probe%s.json"
                       % ("" if d == "rlvs" else "_" + d), required=False)]
    xs = list(range(len(present)))
    for x, d in zip(xs, present):
        j = load(R, "temporal_probe%s.json" % ("" if d == "rlvs" else "_" + d))
        t = j["transforms"]
        n, p = j["n_test"], t["identity"]["accuracy"]
        noise = 100 * j.get("noise_floor_95",
                            1.96 * math.sqrt(p * (1 - p) / n))
        ax.add_patch(plt.Rectangle((x - 0.42, -noise), 0.84, 2 * noise,
                                   facecolor="0.88", edgecolor="none",
                                   zorder=0))
        for k, (key, marker, _) in enumerate(TRANSFORMS):
            ax.plot(x + (k - 1) * 0.17, 100 * t[key]["delta"], marker,
                    color="0.12", markersize=4.5, markerfacecolor="white",
                    markeredgewidth=1.0, zorder=3)

    ax.axhline(0, color="0.45", linewidth=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[d] for d in present], rotation=18,
                       ha="right")
    ax.set_ylabel("change in accuracy (points)")
    ax.set_xlim(-0.65, len(present) - 0.35)
    ax.set_title("(a) frame order destroyed at test time", loc="left")

    handles = [plt.Line2D([], [], marker=m, linestyle="none", color="0.12",
                          markerfacecolor="white", markersize=4.5, label=lab)
               for _, m, lab in TRANSFORMS]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="0.88",
                                 edgecolor="none",
                                 label="95% binomial noise of the split"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7,
              handletextpad=0.5, borderaxespad=0.2, labelspacing=0.35)


def robustness_panel(ax, R):
    j = load(R, "robustness_rlvs.json")
    sev = sorted(set(r["severity"] for r in j["rows"]))

    def series(c):
        return [100 * next(r["accuracy"] for r in j["rows"]
                           if r["corruption"] == c and r["severity"] == s)
                for s in sev]

    for c, style, lab in TEMPORAL:
        ax.plot(sev, series(c), style, color="0.12", markersize=4,
                linewidth=1.3, markerfacecolor="white", label=lab)
    for c, style, lab in APPEARANCE:
        ax.plot(sev, series(c), style, color="0.55", markersize=4,
                linewidth=1.1, markerfacecolor="white", label=lab)

    ax.axhline(50, color="0.75", linewidth=0.7)
    # below the line: above it the rain and low-light curves run into it
    ax.annotate("chance", (sev[1] + 0.5, 50), xytext=(0, -3),
                textcoords="offset points", ha="center", va="top",
                fontsize=7, color="0.55")
    ax.set_xticks(sev)
    ax.set_xlim(sev[0] - 0.35, sev[-1] + 0.35)
    ax.set_xlabel("corruption severity")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(44, 101)
    ax.set_title("(b) the same model under corruption", loc="left")
    # the lower left of this panel is empty, so the legend costs nothing there
    ax.legend(loc="lower left", frameon=False, fontsize=7, handletextpad=0.6,
              borderaxespad=0.2, labelspacing=0.35)


def roc_panel(ax, R):
    """One ROC curve per dataset, pooled over the five seeds.

    The seeds are pooled rather than averaged because the test split is the
    same for all of them, so pooling is just a larger sample of the same
    decision function; averaging curves across seeds would smooth away the
    shape near the top-left corner, which is the part worth seeing.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    greys = ["0.10", "0.30", "0.45", "0.55", "0.65", "0.72"]
    drawn = 0
    for d, st, col in zip(ORDER, styles, greys):
        p = os.path.join(R, "preds_%s.json" % d)
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        y = np.asarray(j["y_true"])
        probs = []
        for v in j["preds"].values():
            # the first round saved hard labels only; later runs save the
            # probability as well, and only the latter can make a curve
            if isinstance(v, dict) and "prob" in v:
                probs.append(np.asarray(v["prob"], float))
        if not probs:
            continue
        yy = np.tile(y, len(probs))
        pp = np.concatenate(probs)
        fpr, tpr, _ = roc_curve(yy, pp)
        ax.plot(fpr, tpr, linestyle=st, color=col, linewidth=1.3,
                label="%s (AUC %.3f)" % (SHORT[d], roc_auc_score(yy, pp)))
        drawn += 1

    if not drawn:
        ax.text(0.5, 0.5, "no probabilities in results/", ha="center",
                va="center", fontsize=8, color="0.4")
        ax.set_axis_off()
        return False

    ax.plot([0, 1], [0, 1], color="0.8", linewidth=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("ROC per dataset, five seeds pooled", loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=7,
              handletextpad=0.6, borderaxespad=0.3, labelspacing=0.35)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="fig")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "text.usetex": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))
    probe_panel(axes[0], args.results)
    robustness_panel(axes[1], args.results)
    fig.tight_layout(w_pad=2.0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        p = "%s_order_invariance.%s" % (args.out, ext)
        fig.savefig(p, bbox_inches="tight")
        print("wrote", p)

    fig2, ax2 = plt.subplots(1, 1, figsize=(3.4, 3.1))
    if roc_panel(ax2, args.results):
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            p = "%s_roc.%s" % (args.out, ext)
            fig2.savefig(p, bbox_inches="tight")
            print("wrote", p)
    else:
        print("skipped the ROC figure: no saved probabilities yet")


if __name__ == "__main__":
    main()

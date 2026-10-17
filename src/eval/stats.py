"""Confidence intervals and significance tests.

Reviewers asked for error bars and a test rather than a single accuracy
number, so results are reported as mean +/- sd over independent seeds, with a
Wilson interval on the pooled test accuracy and McNemar for pairwise model
comparisons on the same test set (Dietterich, 1998).
"""

import itertools

import numpy as np
from scipy import stats


def wilson_interval(n_correct, n_total, confidence=0.95):
    """Wilson score interval - better than the normal approximation near 1.0,
    which matters here because accuracies sit around 0.95."""
    if n_total == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = n_correct / n_total
    denom = 1 + z ** 2 / n_total
    centre = (p + z ** 2 / (2 * n_total)) / denom
    half = z * np.sqrt(p * (1 - p) / n_total +
                       z ** 2 / (4 * n_total ** 2)) / denom
    return (float(centre - half), float(centre + half))


def bootstrap_ci(y_true, y_pred, metric, n_boot=2000, confidence=0.95, seed=0):
    """Percentile bootstrap over the test set, for metrics without a closed
    form (F1, AUC)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(metric(y_true[idx], y_pred[idx]))
        except ValueError:
            continue
    lo = np.percentile(vals, 100 * (1 - confidence) / 2)
    hi = np.percentile(vals, 100 * (1 - (1 - confidence) / 2))
    return float(lo), float(hi)


def mcnemar(y_true, pred_a, pred_b, exact=True):
    """Paired test for two classifiers on the same test set.

    b = A right / B wrong, c = A wrong / B right. With small b+c the exact
    binomial version is used; otherwise the continuity-corrected chi-square.
    """
    y_true = np.asarray(y_true)
    ok_a = np.asarray(pred_a) == y_true
    ok_b = np.asarray(pred_b) == y_true

    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))

    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0,
                "test": "degenerate"}

    if exact and (b + c) < 25:
        p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        return {"b": b, "c": c, "statistic": float(min(b, c)),
                "p_value": float(p), "test": "exact binomial"}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = stats.chi2.sf(chi2, 1)
    return {"b": b, "c": c, "statistic": float(chi2), "p_value": float(p),
            "test": "chi2 with continuity correction"}


def paired_ttest_seeds(scores_a, scores_b):
    """Paired t-test across matched random seeds, plus Cohen's d."""
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    t, p = stats.ttest_rel(a, b)
    diff = a - b
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    return {"t": float(t), "p_value": float(p), "cohens_d": float(d),
            "mean_diff": float(diff.mean())}


def paired_diff_ci(scores_a, scores_b, confidence=0.95):
    """Confidence interval on the mean paired difference across seeds.

    A non-significant p-value says the data are consistent with no difference,
    which is not the same as saying the difference is small. The interval is
    what tells you how large a difference the data still allow.
    """
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    diff = a - b
    n = len(diff)
    if n < 2:
        return {"mean_diff": float(diff.mean()) if n else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n": n}
    se = diff.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    return {"mean_diff": float(diff.mean()),
            "lo": float(diff.mean() - t * se),
            "hi": float(diff.mean() + t * se),
            "se": float(se), "n": int(n), "confidence": confidence}


def tost_equivalence(scores_a, scores_b, margin, alpha=0.05):
    """Two one-sided tests for equivalence within +/- margin.

    Rejecting the two one-sided nulls lets you claim the difference lies inside
    the margin, which an ordinary t-test never licenses however large its
    p-value. The margin has to be chosen before looking, and stated.
    """
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    diff = a - b
    n = len(diff)
    if n < 2:
        return {"equivalent": False, "p_value": float("nan"), "n": n,
                "margin": margin, "reason": "need at least two pairs"}

    se = diff.std(ddof=1) / np.sqrt(n)
    if se == 0:
        # identical on every seed: equivalent for any positive margin
        inside = abs(diff.mean()) < margin
        return {"equivalent": bool(inside), "p_value": 0.0 if inside else 1.0,
                "p_lower": 0.0, "p_upper": 0.0, "mean_diff": float(diff.mean()),
                "margin": float(margin), "n": int(n), "se": 0.0,
                "note": "zero variance across seeds"}

    df = n - 1
    t_lower = (diff.mean() + margin) / se     # H0: diff <= -margin
    t_upper = (diff.mean() - margin) / se     # H0: diff >= +margin
    p_lower = stats.t.sf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p = max(p_lower, p_upper)
    return {"equivalent": bool(p < alpha), "p_value": float(p),
            "p_lower": float(p_lower), "p_upper": float(p_upper),
            "mean_diff": float(diff.mean()), "margin": float(margin),
            "se": float(se), "n": int(n), "alpha": alpha}


def wilcoxon_across_datasets(scores_a, scores_b):
    """Non-parametric comparison over several datasets (Demsar, 2006)."""
    try:
        s, p = stats.wilcoxon(scores_a, scores_b)
    except ValueError:
        return {"statistic": float("nan"), "p_value": float("nan")}
    return {"statistic": float(s), "p_value": float(p)}


def holm_bonferroni(pvalues, alpha=0.05):
    """Step-down correction for a family of comparisons."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adjusted[i] = min(1.0, running)
    return adjusted, adjusted < alpha


def summarise_runs(accs, n_test=None):
    """mean +/- sd over seeds, with a Wilson CI if the test size is known."""
    a = np.asarray(accs, float)
    out = {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1
           else 0.0, "n_runs": int(len(a)),
           "min": float(a.min()), "max": float(a.max())}
    if n_test:
        lo, hi = wilson_interval(int(round(a.mean() * n_test)), n_test)
        out["wilson95"] = (lo, hi)
    return out


def pairwise_mcnemar_table(y_true, predictions):
    """All pairs from {name: preds}, Holm-corrected."""
    names = list(predictions)
    rows, pvals = [], []
    for a, b in itertools.combinations(names, 2):
        r = mcnemar(y_true, predictions[a], predictions[b])
        r["a"], r["b"] = a, b
        rows.append(r)
        pvals.append(r["p_value"])
    adj, sig = holm_bonferroni(pvals)
    for r, pa, s in zip(rows, adj, sig):
        r["p_holm"] = float(pa)
        r["significant"] = bool(s)
    return rows

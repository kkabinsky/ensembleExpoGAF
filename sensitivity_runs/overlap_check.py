# -*- coding: utf-8 -*-
"""
overlap_check.py
================

How much independent information is there, and what does the reported number
actually cost in precision? Reviewer 4, point 4.

Windows advance by one observation and are 32 long, so neighbouring windows
share 31 of their 32 values. The 1,589 scored windows are not 1,589
observations. Saying so is not enough: this program measures what it does to
the numbers.

The measurement needs no retraining. Every detector's per-window score is
already stored, and windows whose starts differ by a multiple of 32 share no
observation at all. Splitting each cell by the start position modulo 32
therefore produces 32 subsets, each internally independent, that between them
use every window exactly once.

    reported     the AUC on the full overlapping set, as in the manuscript
    per offset   the AUC on each of the 32 independent subsets
    spread       the standard deviation across those 32 subsets

The third of these is the honest uncertainty of the first. The full set holds
no more independent information than one subset does, so its AUC carries
roughly the sampling error of a subset, not the much smaller error that its
window count suggests. The ratio between the two is printed for every cell.

The same split shows whether the difference between episodes is real. If
COVID-19 reads well and the Chinese and Iran episodes read poorly, the question
is whether that gap is larger than the gap between two independent subsets of
the same episode.

Run
    python overlap_check.py
    python overlap_check.py --methods hybrid2,tranad,one_class_svm
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "Iran_new_run")
OUT = os.path.join(HERE, "output")

WINDOW = 32
MAPPING = "exponential"
PAPER_METHODS = [
    "anomaly_transformer", "autoencoder", "cnn_autoencoder", "dagmm",
    "deep_svdd", "hybrid2", "isolation_forest", "omnianomaly",
    "one_class_svm", "standalone_fanogan", "tranad", "usad", "vae",
]
EPISODES = {"COVID-19": "results_covid", "Russia-Ukraine": "results_russia",
            "Chinese": "results_chinese", "Iran 2025": "results",
            "Iran 2026": "results_2026"}
ASSETS = ["USOIL", "GOLD", "EURUSD"]


def auc(y, s):
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def hanley_se(a, n1, n0):
    """The standard error an AUC would have if every window were independent.

    This is the figure implied by reporting a window count, and it is the one
    the spread across independent subsets should be compared against.
    """
    if not np.isfinite(a) or n1 < 2 or n0 < 2:
        return float("nan")
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    v = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a))
    return float(np.sqrt(max(v, 0.0) / (n1 * n0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(PAPER_METHODS))
    ap.add_argument("--window", type=int, default=WINDOW)
    a = ap.parse_args()
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for ev, sub in EPISODES.items():
        for asset in ASSETS:
            for m in methods:
                p = os.path.join(RUN, sub, asset, m, MAPPING,
                                 "test_scores.csv")
                if not os.path.isfile(p):
                    continue
                d = pd.read_csv(p).sort_values("window_start")
                st = d["window_start"].to_numpy(int)
                y = d["label_0normal_1crash"].to_numpy(int)
                s = d["anomaly_score"].to_numpy(float)
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                full = auc(y, s)
                se_naive = hanley_se(full, int((y == 1).sum()),
                                     int((y == 0).sum()))
                vals, sizes = [], []
                for off in range(a.window):
                    k = (st % a.window) == off
                    if k.sum() < 4:
                        continue
                    yy = y[k]
                    if yy.sum() == 0 or yy.sum() == len(yy):
                        continue
                    vals.append(auc(yy, s[k]))
                    sizes.append(int(k.sum()))
                if len(vals) < 4:
                    continue
                vals = np.asarray(vals, float)
                rows.append({
                    "event": ev, "asset": asset, "method": m,
                    "n_windows": int(len(y)),
                    "n_independent": int(np.mean(sizes)),
                    "auc_reported": full,
                    "se_if_all_independent": se_naive,
                    "subsets_used": int(len(vals)),
                    "auc_mean_over_subsets": float(vals.mean()),
                    "auc_sd_over_subsets": float(vals.std(ddof=1)),
                    "auc_min": float(vals.min()), "auc_max": float(vals.max()),
                    "inflation": (float(vals.std(ddof=1) / se_naive)
                                  if se_naive and np.isfinite(se_naive)
                                  else float("nan"))})

    t = pd.DataFrame(rows)
    f = os.path.join(OUT, "overlap_check.csv")
    t.round(6).to_csv(f, index=False)
    report(t)
    print()
    print("Per-cell results written to", f)


def report(t):
    print("=" * 84)
    print("What the window count claims, and what the data supports")
    print("=" * 84)
    print("  scored windows in the reported cells            %d"
          % int(t.groupby(["event", "asset"])["n_windows"].first().sum()))
    print("  windows per independent subset, on average      %.1f"
          % t["n_independent"].mean())
    print("  standard error implied by the window count      %.4f"
          % t["se_if_all_independent"].mean())
    print("  spread across independent subsets               %.4f"
          % t["auc_sd_over_subsets"].mean())
    print("  the reported error is understated by a factor of %.1f"
          % t["inflation"].mean())

    print()
    print("=" * 84)
    print("By episode: is the gap between episodes larger than the gap")
    print("between two independent subsets of the same episode?")
    print("=" * 84)
    g = t.groupby("event").agg(
        cells=("auc_reported", "size"),
        auc=("auc_reported", "mean"),
        windows=("n_windows", "mean"),
        independent=("n_independent", "mean"),
        subset_spread=("auc_sd_over_subsets", "mean"),
        worst=("auc_min", "mean"), best=("auc_max", "mean"))
    print(g.round(3).to_string())
    lo, hi = g["auc"].min(), g["auc"].max()
    print()
    print("  spread between episodes            %.3f" % (hi - lo))
    print("  spread within an episode, average  %.3f"
          % g["subset_spread"].mean())
    if (hi - lo) < g["subset_spread"].mean():
        print("  The difference between episodes is smaller than the")
        print("  difference between two independent subsets of one episode.")
    else:
        print("  The difference between episodes exceeds the within-episode")
        print("  subset spread, so it is not explained by the overlap alone.")

    print()
    print("=" * 84)
    print("By detector, reported against the same detector read on")
    print("independent windows only")
    print("=" * 84)
    d = t.groupby("method").agg(
        reported=("auc_reported", "mean"),
        independent_mean=("auc_mean_over_subsets", "mean"),
        subset_spread=("auc_sd_over_subsets", "mean"),
        inflation=("inflation", "mean")).sort_values("reported",
                                                     ascending=False)
    d["gap_to_next"] = d["reported"].diff(-1)
    print(d.round(4).to_string())
    print()
    print("  Compare the gap between two neighbouring detectors with the")
    print("  spread in the column beside it. A ranking whose steps are")
    print("  smaller than that spread is not a ranking.")


if __name__ == "__main__":
    main()

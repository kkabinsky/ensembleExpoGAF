# -*- coding: utf-8 -*-
"""Circular moving-block bootstrap intervals for the core exponential path.

Reproduces the point estimates and 95 per cent intervals reported for the
Iran 2025 and Iran 2026 windows on USOIL, GOLD and EUR/USD.  The interval is
not a normal-approximation interval: adjacent windows advance by one
observation and span 32, so neighbouring rows share most of their input and an
independent-sample interval would be far too narrow.  Resampling contiguous
blocks of ten windows keeps that local dependence inside the resampled unit.

Inputs, both shipped in the repository:

    ensembleExpoGAF/data/iran2025_probabilities_9methods.csv
    ensembleExpoGAF/data/iran2025_hard_predictions_9methods.csv
    ensembleExpoGAF/data/iran2026_probabilities_9methods.csv
    ensembleExpoGAF/data/iran2026_hard_predictions_9methods.csv

AUC is computed from ``prob_EXP``; F1 from ``pred_EXP``, which carries the
0.95-quantile threshold used throughout the paper.  Both files are read for
each interval because each one holds the 200-window slice for its own event.

Protocol: 5,000 replicates, block length 10, seed 42.

Run:

    python bootstrap_ci.py

Writes ``output/bootstrap_ci.csv`` and prints the LaTeX table body.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Iran_new_run", "ensembleExpoGAF", "data")
OUT = os.path.join(HERE, "output")

N_REP = 5000
BLOCK = 10
SEED = 42
ASSETS = ("USOIL", "GOLD", "EURUSD")
INTERVALS = (("2025", "Iran 2025", "iran2025"),
             ("2026", "Iran 2026", "iran2026"))


def auc(y, score):
    """Rank-based AUC.  Returns NaN when one class is absent."""
    y = np.asarray(y)
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return np.nan
    r = pd.Series(score).rank().to_numpy()
    return (r[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg)


def f1(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r)


def block_indices(n, rng):
    """One circular moving-block resample of length n."""
    k = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=k)
    idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel() % n
    return idx[:n]


def interval(y, score, pred, rng):
    n = len(y)
    a, f = [], []
    for _ in range(N_REP):
        i = block_indices(n, rng)
        a.append(auc(y[i], score[i]))
        f.append(f1(y[i], pred[i]))
    a = np.asarray(a, dtype=float)
    f = np.asarray(f, dtype=float)
    a = a[~np.isnan(a)]
    return (np.percentile(f, 2.5), np.percentile(f, 97.5),
            np.percentile(a, 2.5), np.percentile(a, 97.5))


def main():
    rows = []
    for tag, event, stem in INTERVALS:
        prob = pd.read_csv(os.path.join(DATA, stem + "_probabilities_9methods.csv"))
        hard = pd.read_csv(os.path.join(DATA, stem + "_hard_predictions_9methods.csv"))
        for asset in ASSETS:
            p = prob[(prob["event"] == event) & (prob["asset"] == asset)]
            h = hard[(hard["event"] == event) & (hard["asset"] == asset)]
            p = p.sort_values("window_start").reset_index(drop=True)
            h = h.sort_values("window_start").reset_index(drop=True)
            assert len(p) == len(h), "%s %s length mismatch" % (tag, asset)
            assert (p["window_start"].to_numpy()
                    == h["window_start"].to_numpy()).all()
            y = p["label"].to_numpy()
            score = p["prob_EXP"].to_numpy()
            pred = h["pred_EXP"].to_numpy()
            rng = np.random.default_rng(SEED)
            lo_f, hi_f, lo_a, hi_a = interval(y, score, pred, rng)
            rows.append(dict(interval=tag, asset=asset, n=len(y),
                             n_positive=int(y.sum()),
                             f1=f1(y, pred), f1_lo=lo_f, f1_hi=hi_f,
                             auc=auc(y, score), auc_lo=lo_a, auc_hi=hi_a))

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "bootstrap_ci.csv"), index=False)

    print("%-6s %-7s %5s %5s  %-24s %-24s"
          % ("year", "asset", "n", "pos", "F1 [95% CI]", "AUC [95% CI]"))
    for _, r in df.iterrows():
        print("%-6s %-7s %5d %5d  %.3f [%.3f, %.3f]%s  %.3f [%.3f, %.3f]"
              % (r["interval"], r["asset"], r["n"], r["n_positive"],
                 r["f1"], r["f1_lo"], r["f1_hi"], " " * 5,
                 r["auc"], r["auc_lo"], r["auc_hi"]))

    print()
    print("LaTeX table body:")
    last = None
    for _, r in df.iterrows():
        head = r["interval"] if r["interval"] != last else " " * 4
        last = r["interval"]
        print("%s & %-6s & %.3f [%.3f, %.3f] & %.3f [%.3f, %.3f] \\\\"
              % (head, r["asset"], r["f1"], r["f1_lo"], r["f1_hi"],
                 r["auc"], r["auc_lo"], r["auc_hi"]))


if __name__ == "__main__":
    main()

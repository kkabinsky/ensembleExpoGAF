# -*- coding: utf-8 -*-
"""Diebold--Mariano matrix corrected for overlapping windows.

The matrix as published divides the mean loss differential by the plain sample
standard deviation.  Windows advance by one observation and span 32, so they
share 31 of their 32 values and the loss differential is strongly
autocorrelated; that divisor understates the standard error and inflates every
statistic.  This script recomputes the same 45 pairwise comparisons on the same
pooled windows with a Newey--West long-run variance at lag 31, then applies the
Holm step-down adjustment across the 45 pairs.

Inputs, both shipped in the repository:

    ensembleExpoGAF/data/aligned_probabilities_9methods.csv   loss_<M> per window
    ensembleExpoGAF/data/aligned_hard_predictions_10methods.csv  loss_ENS

Run:

    python dm_overlap_corrected.py

Writes ``output/dm_overlap_corrected.csv`` with one row per ordered pair, and
prints the LaTeX body of the corrected matrix.
"""
import itertools
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Iran_new_run", "ensembleExpoGAF", "data")
OUT = os.path.join(HERE, "output")

LAG = 31
ALPHA = 0.05

# row order as printed in the manuscript
METHODS = [("ENS", "EnsembleExpoGAF soft vote (ENS)"),
           ("EXP", "ExpoGAF-AnoNet (EXP)"),
           ("AT", "Anomaly Transformer (AT)"),
           ("DAG", "DAGMM (DAG)"),
           ("DSV", "Deep SVDD (DSV)"),
           ("IF", "Isolation Forest (IF)"),
           ("OMNI", "OmniAnomaly (OMNI)"),
           ("OCS", "One-Class SVM (OCS)"),
           ("TR", "TranAD (TR)"),
           ("USAD", "USAD (USAD)")]


def newey_west_var(d, lag):
    """Long-run variance of the mean of d, Bartlett kernel."""
    n = len(d)
    x = d - d.mean()
    g0 = float(x @ x) / n
    s = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(x[k:] @ x[:-k]) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return max(s, 1e-12) / n


def normal_two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def holm(pvals):
    """Holm step-down adjusted p values, order preserved."""
    m = len(pvals)
    idx = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for r, i in enumerate(idx):
        v = (m - r) * pvals[i]
        run = max(run, v)
        adj[i] = min(1.0, run)
    return adj


def main():
    prob = pd.read_csv(os.path.join(DATA, "aligned_probabilities_9methods.csv"))
    hard = pd.read_csv(os.path.join(DATA,
                                    "aligned_hard_predictions_10methods.csv"))
    key = ["event", "asset", "window_start"]
    df = prob.merge(hard[key + ["loss_ENS"]], on=key, how="inner")
    df = df.sort_values(key).reset_index(drop=True)

    cols = {m: "loss_" + m for m, _ in METHODS}
    have = [m for m, _ in METHODS if cols[m] in df.columns]
    missing = [m for m, _ in METHODS if cols[m] not in df.columns]
    if missing:
        raise SystemExit("missing loss columns: %s" % ", ".join(missing))

    sub = df[[cols[m] for m in have]].dropna()
    n = len(sub)
    print("pooled windows common to all ten methods: %d" % n)
    print("Newey--West lag %d, Holm across the 45 unique pairs" % LAG)
    print()

    loss = {m: sub[cols[m]].to_numpy(dtype=float) for m in have}

    pairs = list(itertools.combinations(have, 2))
    stats, raw_p = {}, []
    for a, b in pairs:
        # d_t = L(column) - L(row) with row=a, column=b; positive favours a
        d = loss[b] - loss[a]
        var = newey_west_var(d, LAG)
        z = d.mean() / math.sqrt(var)
        p = normal_two_sided(z)
        stats[(a, b)] = z
        raw_p.append(p)

    adj = holm(raw_p)
    padj = {pr: adj[i] for i, pr in enumerate(pairs)}
    praw = {pr: raw_p[i] for i, pr in enumerate(pairs)}

    rows = []
    for (a, b) in pairs:
        rows.append(dict(row=a, column=b, dm=stats[(a, b)],
                         p_raw=praw[(a, b)], p_holm=padj[(a, b)],
                         significant=padj[(a, b)] < ALPHA))
    res = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "dm_overlap_corrected.csv"), index=False)

    surv = res[res["significant"]]
    print("pairs significant after the adjustment: %d of %d"
          % (len(surv), len(res)))
    for _, r in surv.iterrows():
        print("   %-5s vs %-5s  DM %+7.2f  adjusted p %.4g"
              % (r["row"], r["column"], r["dm"], r["p_holm"]))

    print()
    print("LaTeX body of the corrected matrix:")
    name = dict(METHODS)
    for a in have:
        cells = []
        for b in have:
            if a == b:
                cells.append("$0.00$")
                continue
            if (a, b) in stats:
                z, p = stats[(a, b)], padj[(a, b)]
            else:
                z, p = -stats[(b, a)], padj[(b, a)]
            body = "%+.2f" % z
            cells.append("$\\mathbf{%s}^{*}$" % body if p < ALPHA
                         else "$%s$" % body)
        print("%s & %s \\\\" % (name[a], " & ".join(cells)))


if __name__ == "__main__":
    main()

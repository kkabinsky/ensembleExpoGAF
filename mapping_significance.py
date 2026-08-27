# -*- coding: utf-8 -*-
"""
mapping_significance.py
=======================

The mapping ordering changes on the corrected pipeline and the exponential map
moves to first place. This script asks whether that lead is large enough to
report as a difference, or whether it is within sampling variation.

Three tests, because they answer different questions:

1. Diebold-Mariano on per-window Brier loss, with a Newey-West correction at
   lag 31 for the overlap between adjacent windows, then Holm correction over
   the three comparisons.
2. A paired sign test with the episode-asset cell as the unit, which is the
   conservative reading given how little independent information the windows
   carry.
3. Friedman across all four mappings, to ask whether they differ at all as a
   set.

Run
    python mapping_significance.py
"""
import io
import os
import sys
from math import comb, erfc, sqrt

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "Iran_new_run")
EPISODES = {"Iran 2025": "results", "COVID-19": "results_covid",
            "Russia-Ukraine": "results_russia", "Chinese": "results_chinese",
            "Iran 2026": "results_2026"}
ASSETS = ("USOIL", "GOLD", "EURUSD")
MAPS = ("cosine", "arctan", "arccosh", "exponential")
ARM = "hybrid_true"          # the corrected arm: GAF of TadGAN scores
WINDOW = 32
LAG = WINDOW - 1


def auc(y, s):
    y = np.asarray(y, int)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def rank01(s):
    r = pd.Series(s).rank(method="average").to_numpy()
    return (r - 0.5) / len(r)


def nw_var(d, lag):
    d = np.asarray(d, float)
    d = d - d.mean()
    n = len(d)
    g0 = np.dot(d, d) / n
    v = g0
    for k in range(1, lag + 1):
        if k >= n:
            break
        w = 1.0 - k / (lag + 1.0)
        gk = np.dot(d[k:], d[:-k]) / n
        v += 2 * w * gk
    return v / n


def dm(la, lb, lag):
    d = np.asarray(la, float) - np.asarray(lb, float)
    v = nw_var(d, lag)
    if v <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / sqrt(v)
    p = erfc(abs(stat) / sqrt(2))
    return stat, p


# read the scores and labels of every labelled cell
loss = {m: [] for m in MAPS}     # Brier loss pooled over all windows
cell_auc = {m: {} for m in MAPS}
for ep, sub in EPISODES.items():
    for a in ASSETS:
        base = {}
        okcell = True
        for m in MAPS:
            f = os.path.join(RUN, sub, a, ARM, m, "test_scores.csv")
            if not os.path.isfile(f):
                okcell = False
                break
            d = pd.read_csv(f)
            y = d["label_0normal_1crash"].to_numpy(int)
            s = d["anomaly_score"].to_numpy(float)
            base[m] = (y, s)
        if not okcell or base[MAPS[0]][0].sum() == 0:
            continue
        for m in MAPS:
            y, s = base[m]
            p = rank01(s)
            loss[m].append((p - y) ** 2)
            cell_auc[m][(ep, a)] = auc(y, s)

for m in MAPS:
    loss[m] = np.concatenate(loss[m]) if loss[m] else np.array([])

ncells = len(cell_auc["exponential"])
nwin = len(loss["exponential"])
print("Labelled cells %d, windows in total %d" % (ncells, nwin))
print("Independent observations about %d/32 = %.0f" % (nwin, nwin / 32))

print()
print("=" * 78)
print("1. Diebold-Mariano on Brier loss, exponential against the other three")
print("=" * 78)
print("   d_t = loss(other) - loss(exponential); positive favours exponential")
pvals = []
for m in MAPS:
    if m == "exponential":
        continue
    st0, p0 = dm(loss[m], loss["exponential"], 0)
    st31, p31 = dm(loss[m], loss["exponential"], LAG)
    pvals.append(p31)
    print("   exp vs %-8s  loss %.4f vs %.4f   DM lag0 %+.3f (p %.4f)   "
          "lag31 %+.3f (p %.4f)"
          % (m, loss["exponential"].mean(), loss[m].mean(),
             st0, p0, st31, p31))
# Holm correction over the three comparisons
order = np.argsort(pvals)
holm = [False] * 3
adj = [0.0] * 3
for rank, idx in enumerate(order):
    adj[idx] = min(1.0, pvals[idx] * (3 - rank))
names = [m for m in MAPS if m != "exponential"]
print()
print("   after Holm correction over three pairs:")
for i, m in enumerate(names):
    print("     exp vs %-8s  p_holm %.4f  %s"
          % (m, adj[i], "significant" if adj[i] < 0.05 else "not significant"))

print()
print("=" * 78)
print("2. Paired sign test, with the cell as the unit")
print("=" * 78)
for m in names:
    w = l = 0
    for k in cell_auc["exponential"]:
        de = cell_auc["exponential"][k] - cell_auc[m][k]
        if de > 0:
            w += 1
        elif de < 0:
            l += 1
    n = w + l
    p = (min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)
         if n else float("nan"))
    print("   exp vs %-8s  exp wins %d loses %d of %d cells   p %.4f"
          % (m, w, l, n, p))

print()
print("=" * 78)
print("3. Friedman: do the four differ as a set, with the cell as the unit")
print("=" * 78)
cells = sorted(cell_auc["exponential"])
mat = np.array([[cell_auc[m][k] for m in MAPS] for k in cells])
ranks = np.array([pd.Series(-row).rank(method="average").to_numpy()
                  for row in mat])
n, k = ranks.shape
Rj = ranks.sum(axis=0)
chi = 12.0 / (n * k * (k + 1)) * np.sum(Rj ** 2) - 3 * n * (k + 1)
from math import gamma
def chi2_sf(x, df):
    # simple numerical integration, accurate enough to report
    import math
    if x <= 0:
        return 1.0
    # series expansion of the regularized lower incomplete gamma
    a = df / 2.0
    xx = x / 2.0
    term = 1.0 / a
    s = term
    for i in range(1, 200):
        term *= xx / (a + i)
        s += term
    low = s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    return max(0.0, 1.0 - low)
p = chi2_sf(chi, k - 1)
print("   mean rank per mapping (lower is better):")
for j, m in enumerate(MAPS):
    print("     %-12s %.2f" % (m, Rj[j] / n))
print("   chi2 = %.3f  df = %d  p = %.4f  %s"
      % (chi, k - 1, p, "the four differ" if p < 0.05 else "the four are not separated"))

print()
print("=" * 78)
print("Summary")
print("=" * 78)
print("  exponential has the highest mean AUC and the best mean rank here")
print("  but check the tests above before calling it a win")
print("  if they do not pass, report it as best on average but not separated")

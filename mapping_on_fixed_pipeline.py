# -*- coding: utf-8 -*-
"""
mapping_on_fixed_pipeline.py
============================

Compare the four angular mappings on the corrected pipeline.

Why this exists
    The mapping ordering the manuscript reports came from the earlier pipeline,
    in which the two arms carried identical scores in all 72 cells and therefore
    compared nothing. The corrected runner trains both arms in one round with
    identical settings. The question is whether the ordering survives once the
    comparison is real.

Only cells finished on both arms are read; unfinished cells are skipped, so the
script can be run while a sweep is still in progress and will show what exists.

Run
    python mapping_on_fixed_pipeline.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "Iran_new_run")
EPISODES = {"Iran 2025": "results", "COVID-19": "results_covid",
            "Russia-Ukraine": "results_russia", "Chinese": "results_chinese",
            "Iran 2026": "results_2026",
            "Normal 2019": os.path.join("covid_normal", "results")}
ASSETS = ("USOIL", "GOLD", "EURUSD")
MAPS = ("cosine", "arctan", "arccosh", "exponential")
ARMS = ("hybrid_true", "standalone_ref")


def auc(y, s):
    y = np.asarray(y, int)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


rows = []
for ep, sub in EPISODES.items():
    for a in ASSETS:
        for m in MAPS:
            fs = {arm: os.path.join(RUN, sub, a, arm, m, "test_scores.csv")
                  for arm in ARMS}
            if not all(os.path.isfile(f) for f in fs.values()):
                continue
            rec = {"episode": ep, "asset": a, "mapping": m}
            ok = True
            for arm, f in fs.items():
                d = pd.read_csv(f)
                y = d["label_0normal_1crash"].to_numpy(int)
                s = d["anomaly_score"].to_numpy(float)
                rec["auc_" + arm] = auc(y, s)
                rec["n"] = len(d)
                rec["n_pos"] = int(y.sum())
            if ok:
                rows.append(rec)

if not rows:
    sys.exit("No cell has both arms finished yet")

t = pd.DataFrame(rows)
t["d_auc"] = t["auc_hybrid_true"] - t["auc_standalone_ref"]

print("Cells finished on both arms: %d" % len(t))
print("Episodes complete across all four mappings:")
full = []
for ep in t["episode"].unique():
    for a in ASSETS:
        sub = t[(t["episode"] == ep) & (t["asset"] == a)]
        if len(sub) == len(MAPS):
            full.append((ep, a))
for ep, a in full:
    print("   %s %s" % (ep, a))
if not full:
    print("   none yet")
    sys.exit()

print()
print("=" * 84)
print("AUC of the corrected arm, GAF of TadGAN scores, by cell")
print("=" * 84)
lab = t[t["auc_hybrid_true"].notna()]
p = lab.pivot_table(index=["episode", "asset"], columns="mapping",
                    values="auc_hybrid_true")
p = p[[m for m in MAPS if m in p.columns]]
p = p.dropna()
if p.empty:
    print("  no labelled cell is complete across all four mappings yet")
else:
    p2 = p.copy()
    p2["best"] = p.idxmax(axis=1)
    print(p2.round(4).to_string())

    print()
    print("=" * 84)
    print("Mean rank per mapping, 1 = best, over cells complete in all four")
    print("=" * 84)
    rank = p.rank(axis=1, ascending=False)
    for m in p.columns:
        n1 = int((p.idxmax(axis=1) == m).sum())
        print("  %-12s mean AUC %.4f   mean rank %.2f   first in %d of %d cells"
              % (m, p[m].mean(), rank[m].mean(), n1, len(p)))

    print()
    print("=" * 84)
    print("Against the ordering the manuscript reports from the earlier pipeline")
    print("=" * 84)
    if len(p) < 12:
        print("  WARNING  only %d cells so far, of the 15 labelled cells" % len(p))
        print("         the ordering shown is mostly noise; do not quote it")
        print("         wait for the sweep to finish, then run this again")
        print()
    OLD = {"arccosh": 0.5276, "arctan": 0.5238, "cosine": 0.5217,
           "exponential": 0.4849}
    old_order = sorted(OLD, key=OLD.get, reverse=True)
    new_order = list(p.mean().sort_values(ascending=False).index)
    print("  earlier pipeline   %s" % "  >  ".join(old_order))
    print("  corrected pipeline %s" % "  >  ".join(new_order))
    print()
    if old_order == new_order:
        print("  ordering unchanged")
    else:
        print("  ordering changed; this must be reported to the reviewers")

print()
print("=" * 84)
print("Gap between the two arms by mapping, i.e. what the TadGAN stage adds")
print("=" * 84)
g = t.groupby("mapping")["d_auc"].agg(["mean", "std", "size"])
print(g.round(4).to_string())

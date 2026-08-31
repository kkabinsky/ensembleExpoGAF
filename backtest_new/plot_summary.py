# -*- coding: utf-8 -*-
"""
plot_summary.py
===============

Draw the two-panel summary figure from the backtest output.

Output

output/backtest_summary.jpg

output/backtest_summary.pdf


Run
    python plot_summary.py
    python plot_summary.py --mode far0.10
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

ORDER = ["IF", "OCS", "DSV", "DAG", "OMNI", "USAD", "TR", "AT", "EXP", "ENS",
         "ENST"]
LABEL = {"IF": "Isolation Forest", "OCS": "One-Class SVM", "DSV": "Deep SVDD",
         "DAG": "DAGMM", "OMNI": "OmniAnomaly", "USAD": "USAD",
         "TR": "TranAD", "AT": "Anomaly Transformer", "EXP": "ExpoGAF-AnoNet",
         "ENS": "Majority vote", "ENST": "EnsembleExpoGAF"}
SHORT = {"IF": "IF", "OCS": "OCS", "DSV": "DSV", "DAG": "DAG",
         "OMNI": "OMNI", "USAD": "USAD", "TR": "TR", "AT": "AT",
         "EXP": "EXP", "ENS": "ENS-H", "ENST": "ENS"}
# a few labels sit on top of each other at the default offset; these nudge
# them apart and change nothing else on the figure
NUDGE = {"IF": (7, -12), "OMNI": (7, 6), "AT": (-4, 10), "ENST": (8, -4),
         "DAG": (7, 5), "EXP": (6, -13)}
BASE_C = "#6f8da2"
EXP_C = "#cf4358"
ENS_C = "#7b4ea3"
# The legend states what each rule does rather than what it is called. The
# manuscript calls the first one the Defensive strategy, but a reader meeting
# that word in a legend cannot tell what it means, and the earlier wording,
# "published rule", read as though it concerned publication.
RULES = [("defensive", "Cash on any alarm day"),
         ("first_alarm", "Cash from the first alarm on")]


def colour(m):
    return {"EXP": EXP_C, "ENST": ENS_C, "ENS": "#18895e"}.get(m, BASE_C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="as reported")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    p = os.path.join(OUT, "backtest_new_cells.csv")
    if not os.path.isfile(p):
        raise SystemExit("run backtest_new.py first; %s is missing" % p)
    d = pd.read_csv(p)
    d = d[(d["event"] != "Normal 2019") & (d["mode"] == a.mode)]
    if d.empty:
        raise SystemExit("no rows for mode %s" % a.mode)
    present = [m for m in ORDER if m in set(d["method"])]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6),
                             gridspec_kw={"width_ratios": [1.12, 1]})

    # ---- left: mean excess return, two rules side by side ----
    ax = axes[0]
    order = (d[d["strategy"] == "first_alarm"].groupby("method")["delta_pct"]
             .mean().reindex(present).sort_values(ascending=True))
    names = list(order.index)
    y = np.arange(len(names))
    height = 0.38
    for k, (rule, _) in enumerate(RULES):
        s = d[d["strategy"] == rule]
        mu = s.groupby("method")["delta_pct"].mean().reindex(names)
        n = s.groupby("method")["delta_pct"].size().reindex(names)
        se = (s.groupby("method")["delta_pct"].std().reindex(names)
              / np.sqrt(n.clip(lower=1)))
        off = (0.5 - k) * height
        ax.barh(y + off, mu.to_numpy(), height=height,
                color=[colour(m) for m in names],
                alpha=1.0 if rule == "first_alarm" else 0.45,
                edgecolor="white", linewidth=0.6,
                xerr=se.to_numpy(), error_kw=dict(ecolor="#34495e", lw=0.9,
                                                  capsize=2.5))
    ax.axvline(0.0, color="#303030", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in names], fontsize=9)
    ax.set_xlabel("Mean return above Buy-and-Hold, per cent", fontsize=9.5)
    ax.grid(axis="x", alpha=0.28, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#7f8b93")
    ax.set_title("(a)", fontsize=11, weight="bold", loc="left")
    ax.legend(handles=[
        Patch(facecolor="#6f8da2", alpha=0.45, label=RULES[0][1]),
        Patch(facecolor="#6f8da2", label=RULES[1][1])],
        loc="lower right", frameon=False, fontsize=8)

    # ---- right: timing against exposure ----
    ax = axes[1]
    s = d[d["strategy"] == "first_alarm"]
    g = s.groupby("method").agg(nul=("null_pct", "mean"),
                                inm=("days_in_market", "mean")).reindex(names)
    for m in names:
        ax.scatter(g.loc[m, "inm"], g.loc[m, "nul"],
                   s=190 if m in ("EXP", "ENST") else 95,
                   color=colour(m), edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(SHORT[m], (g.loc[m, "inm"], g.loc[m, "nul"]),
                    textcoords="offset points",
                    xytext=NUDGE.get(m, (7, 5)), fontsize=8.5,
                    fontweight="bold" if m in ("EXP", "ENST") else "normal")
    ax.axhline(0.5, color="#303030", lw=1.1, ls="--")
    ax.text(0.99, 0.5, " random", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, style="italic", color="#555")
    ax.set_xlabel("Share of days holding the asset", fontsize=9.5)
    ax.set_ylabel("Percentile against random exit dates", fontsize=9.5)
    ax.grid(alpha=0.28, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("(b)", fontsize=11, weight="bold", loc="left")

    v = s.groupby("method")["delta_pct"].mean().reindex(names)
    r = float(np.corrcoef(g["inm"].to_numpy(float), v.to_numpy(float))[0, 1])
    fig.text(0.5, 0.022,
             "Fifteen crisis cells.  Error bars: standard error across cells."
             "  Correlation between (a) and the exposure in (b): %+.2f." % r,
             ha="center", fontsize=8.5, color="#61707c")
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.14, top=0.92,
                        wspace=0.32)

    stem = os.path.join(OUT, "backtest_summary")
    fig.savefig(stem + ".jpg", dpi=a.dpi, bbox_inches="tight",
                pil_kwargs={"quality": 95})
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print("written")
    print("  ", stem + ".jpg")
    print("  ", stem + ".pdf")
    print()
    print("Mean above Buy-and-Hold, per cent, thresholds %s" % a.mode)
    print("%-22s %12s %12s %10s %10s"
          % ("method", RULES[0][0], RULES[1][0], "in market", "beats random"))
    for m in reversed(names):
        row = []
        for rule, _ in RULES:
            row.append(d[(d["strategy"] == rule)
                         & (d["method"] == m)]["delta_pct"].mean())
        print("%-22s %12.2f %12.2f %10.3f %10.3f"
              % (LABEL[m], row[0], row[1], g.loc[m, "inm"], g.loc[m, "nul"]))
    print()
    print("correlation between excess return and time in the market %+.3f" % r)


if __name__ == "__main__":
    main()

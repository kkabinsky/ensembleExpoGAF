# -*- coding: utf-8 -*-
"""
plot_backtest_new.py
====================

The bar charts of the manuscript redrawn from `backtest_new.py`, in the layout
of the published figures: one panel per asset, one bar per detector, the bar
height the final cumulative return over the test window, the value printed
above each bar, and the endpoint count and date range above each panel.

What differs from the published figures is the rule, and it is stated on the
figure itself.

Outputs, all under output/
    backtest_bar_<event>_new.png and .pdf    one figure per episode
    iran_2025_2026_<rule>_6panels_new.pdf    the replacement for Figure 3

Run
    python plot_backtest_new.py
    python plot_backtest_new.py --strategy defensive --mode "as reported"
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

ASSETS = ["USOIL", "GOLD", "EURUSD"]
ORDER = ["IF", "OCS", "DSV", "DAG", "OMNI", "USAD", "TR", "AT", "EXP",
         "ENS", "ENST"]
LABEL = {"IF": "IF", "OCS": "OCS", "DSV": "DSV", "DAG": "DAG", "OMNI": "OMNI",
         "USAD": "USAD", "TR": "TR", "AT": "AT", "EXP": "EXP", "ENS": "ENS-H",
         "ENST": "ENS"}
# ENS-H, the unweighted majority alarm, is only drawn when backtest_new.py was
# run with --keep-majority; otherwise it is absent from the input file and the
# bar simply does not appear.
EVENT_LABEL = {"COVID-19": "COVID-19", "Russia--Ukraine": "Russia-Ukraine",
               "Chinese": "Chinese real-estate stress",
               "Iran 2025": "Iran 2025", "Iran 2026": "Iran 2026 snapshot",
               "Normal 2019": "Normal 2019"}
RULE_TITLE = {"defensive": "alarm-index Defensive backtest",
              "first_alarm": "first-alarm backtest",
              "persist": "persistence backtest"}
RULE_NOTE = {
    "defensive": "An alarm at t moves the strategy to cash for t+1; cash "
                 "earns 0%.",
    "first_alarm": "The position is held until the first alarm and is in cash "
                   "from then to the end of the window; cash earns 0%.",
    "persist": "The position leaves after two alarms in a row and returns "
               "after five clear days; cash earns 0%."}

BASE_C = "#6f8da2"
EXP_C = "#cf4358"
ENS_C = "#18895e"
ENST_C = "#7b4ea3"


def colour(m):
    return {"EXP": EXP_C, "ENS": ENS_C, "ENST": ENST_C}.get(m, BASE_C)


def limits(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return -1.0, 1.0
    lo, hi = min(v.min(), 0.0), max(v.max(), 0.0)
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return lo - 0.16 * span, hi + 0.22 * span


def draw_panel(ax, cell, show_ylabel):
    cell = cell.set_index("method").reindex(
        [m for m in ORDER if m in set(cell["method"])])
    v = cell["final_return_pct"].to_numpy(float)
    x = np.arange(len(cell))
    bars = ax.bar(x, v, color=[colour(m) for m in cell.index],
                  edgecolor="none", width=0.70)
    ax.axhline(0.0, color="#303030", linewidth=0.9)
    ax.set_ylim(*limits(v))
    if show_ylabel:
        ax.set_ylabel("Final return (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in cell.index], fontsize=8.5)
    ax.grid(axis="y", alpha=0.28, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#7f8b93")
    bottom, top = ax.get_ylim()
    off = 0.018 * (top - bottom)
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2,
                val + off if val >= 0 else val - off, "%.1f" % val,
                va="bottom" if val >= 0 else "top", ha="center",
                fontsize=7.4, color="#202020")


def panel_heading(ax, cell, title):
    ax.text(0.0, 1.13, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11.5, weight="bold", color="#111111")
    r = cell.iloc[0]
    ax.text(0.0, 1.065, "%d endpoints: %s to %s"
            % (int(r["n_endpoints"]), r["start_date"], r["end_date"]),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
            color="#61707c")


def legend_handles(has_majority):
    h = [Patch(facecolor=BASE_C, label="Eight baselines"),
         Patch(facecolor=EXP_C, label="ExpoGAF-AnoNet"),
         Patch(facecolor=ENST_C, label="EnsembleExpoGAF")]
    if has_majority:
        h.insert(2, Patch(facecolor=ENS_C, label="Majority vote (5 of 9)"))
    return h


def footnotes(fig, rule, mode, y1, y2):
    fig.text(0.5, y1, RULE_NOTE[rule] + "  Thresholds: %s." % mode,
             ha="center", fontsize=8.5, color="#61707c")
    fig.text(0.5, y2,
             "IF: Isolation Forest; OCS: One-Class SVM; DSV: Deep SVDD; "
             "DAG: DAGMM; OMNI: OmniAnomaly; TR: TranAD; "
             "AT: Anomaly Transformer; EXP: ExpoGAF-AnoNet; "
             "ENS: EnsembleExpoGAF.",
             ha="center", fontsize=7.5, color="#61707c")


def draw_event(d, event, rule, mode):
    has_maj = "ENS" in set(d["method"])
    sub = d[d["event"] == event]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8))
    for ax, asset in zip(axes, ASSETS):
        cell = sub[sub["asset"] == asset]
        if cell.empty:
            ax.set_visible(False)
            continue
        draw_panel(ax, cell, asset == "USOIL")
        panel_heading(ax, cell, "%s - %s" % (EVENT_LABEL[event], asset))
    fig.suptitle("%s: %s" % (EVENT_LABEL[event], RULE_TITLE[rule]),
                 fontsize=14, weight="bold", y=0.995)
    fig.legend(handles=legend_handles(has_maj), loc="upper center",
               bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False, fontsize=9)
    footnotes(fig, rule, mode, 0.035, 0.012)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.15, top=0.76,
                        wspace=0.18)
    stem = os.path.join(OUT, "backtest_bar_%s_new"
                        % event.replace("--", "_").replace(" ", "").replace(
                            "-", ""))
    fig.savefig(stem + ".png", dpi=240, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return stem


def draw_six(d, rule, mode):
    """The replacement for Figure 3: Iran 2025 and Iran 2026, three assets."""
    has_maj = "ENS" in set(d["method"])
    events = ["Iran 2025", "Iran 2026"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.4))
    for row, ev in enumerate(events):
        for col, asset in enumerate(ASSETS):
            ax = axes[row][col]
            cell = d[(d["event"] == ev) & (d["asset"] == asset)]
            if cell.empty:
                ax.set_visible(False)
                continue
            draw_panel(ax, cell, asset == "USOIL")
            panel_heading(ax, cell, "%s - %s" % (EVENT_LABEL[ev], asset))
    fig.suptitle("Iran 2025 and June 2026: %s" % RULE_TITLE[rule],
                 fontsize=14, weight="bold", y=0.995)
    fig.legend(handles=legend_handles(has_maj), loc="upper center",
               bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False, fontsize=9)
    footnotes(fig, rule, mode, 0.028, 0.010)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.09, top=0.87,
                        wspace=0.18, hspace=0.42)
    stem = os.path.join(OUT, "iran_2025_2026_%s_6panels_new" % rule)
    fig.savefig(stem + ".png", dpi=240, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="first_alarm",
                    choices=["defensive", "first_alarm", "persist"])
    ap.add_argument("--mode", default="as reported")
    a = ap.parse_args()

    p = os.path.join(OUT, "backtest_new_cells.csv")
    if not os.path.isfile(p):
        raise SystemExit("run backtest_new.py first; %s is missing" % p)
    d = pd.read_csv(p)
    d = d[(d["strategy"] == a.strategy) & (d["mode"] == a.mode)]
    if d.empty:
        raise SystemExit("no rows for strategy %s and mode %s"
                         % (a.strategy, a.mode))

    print("rule %s, thresholds %s" % (a.strategy, a.mode))
    for ev in d["event"].unique():
        s = draw_event(d, ev, a.strategy, a.mode)
        if s:
            print("  ", os.path.basename(s) + ".pdf")
    s = draw_six(d, a.strategy, a.mode)
    print("  ", os.path.basename(s) + ".pdf   (the replacement for Figure 3)")

    print()
    print("Bar heights, mean over the fifteen crisis cells")
    g = d[d["event"] != "Normal 2019"].groupby("method")["delta_pct"].mean()
    for m in ORDER:
        if m in g.index:
            print("  %-6s %+7.2f above buy and hold" % (LABEL[m], g[m]))


if __name__ == "__main__":
    main()

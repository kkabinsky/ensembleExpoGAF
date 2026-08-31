# -*- coding: utf-8 -*-
"""
pr_auc_all_datasets_new.py

Average precision for every method on every asset-episode cell.

Run

    python pr_auc_all_datasets_new.py
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "ensembleExpoGAF" / "data"

METHODS = [
    ("ENS",  "EnsembleExpoGAF"),
    ("EXP",  "ExpoGAF-AnoNet (core)"),
    ("AT",   "Anomaly Transformer"),
    ("DAG",  "DAGMM"),
    ("DSV",  "Deep SVDD"),
    ("IF",   "Isolation Forest"),
    ("OMNI", "OmniAnomaly"),
    ("OCS",  "One-Class SVM"),
    ("TR",   "TranAD"),
    ("USAD", "USAD"),
]
EVENT_ORDER = ["COVID-19", "Russia--Ukraine", "Chinese", "Iran 2025", "Iran 2026"]
ASSET_ORDER = ["USOIL", "GOLD", "EURUSD"]


def load():
    p = pd.read_csv(DATA / "aligned_probabilities_9methods.csv")
    h = pd.read_csv(DATA / "aligned_hard_predictions_10methods.csv")
    keys = ["event", "asset", "window_start"]
    if not p[keys].equals(h[keys]):
        sys.exit("the two files are not aligned row for row")
    d = p.copy()
    d["prob_ENS"] = h["prob_ENS"].to_numpy(float)
    # Normal 2019 holds no positive window, so AP is undefined there
    return d[d["event"] != "Normal 2019"].reset_index(drop=True)


def main():
    d = load()
    rows = []
    for (ev, asset), g in d.groupby(["event", "asset"], sort=False):
        y = g["label"].to_numpy(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        prev = float(y.mean())
        rec = dict(event=ev, asset=asset, n=len(y), n_pos=int(y.sum()),
                   prevalence=prev)
        for k, _lab in METHODS:
            ap = float(average_precision_score(y, g["prob_" + k].to_numpy(float)))
            rec["AP_" + k] = ap
            rec["lift_" + k] = ap - prev
        rows.append(rec)
    cells = pd.DataFrame(rows)
    cells["event"] = pd.Categorical(cells["event"], EVENT_ORDER, ordered=True)
    cells["asset"] = pd.Categorical(cells["asset"], ASSET_ORDER, ordered=True)
    cells = cells.sort_values(["event", "asset"]).reset_index(drop=True)
    cells.round(6).to_csv(HERE / "pr_auc_all_cells_new.csv", index=False)

    ap_cols = ["AP_" + k for k, _ in METHODS]
    best = cells[ap_cols].idxmax(axis=1).str.replace("AP_", "", regex=False)
    cells_out = cells.copy()
    cells_out["best_method"] = best

    print("=" * 96)
    print("PR-AUC by cell, 15 cells; the bracket is the lift over random")
    print("=" * 96)
    hdr = "%-16s %-7s %5s %6s  " % ("event", "asset", "n", "prev")
    hdr += "".join("%-16s" % k for k, _ in METHODS)
    print(hdr)
    for _, r in cells_out.iterrows():
        line = "%-16s %-7s %5d %6.3f  " % (r["event"], r["asset"], r["n"],
                                           r["prevalence"])
        for k, _ in METHODS:
            mark = "*" if r["best_method"] == k else " "
            line += "%s%.3f(%+.3f)%s" % (mark, r["AP_" + k], r["lift_" + k],
                                         "" if mark == "*" else " ")
        print(line)
    print()
    print("  * marks the highest AP in that cell")
    print()

    # -------------------------------------------------- summary by method
    srows = []
    for k, lab in METHODS:
        ap = cells["AP_" + k]
        lift = cells["lift_" + k]
        rank = cells[ap_cols].rank(axis=1, ascending=False)["AP_" + k]
        srows.append(dict(
            key=k, method=lab, cells=len(cells),
            mean_AP=ap.mean(), sd_AP=ap.std(),
            mean_lift=lift.mean(),
            cells_above_random=int((lift > 0).sum()),
            cells_best=int((best == k).sum()),
            mean_rank=rank.mean(), best_rank=int(rank.min()),
            worst_rank=int(rank.max())))
    summ = pd.DataFrame(srows).sort_values("mean_AP", ascending=False)
    summ.round(6).to_csv(HERE / "pr_auc_summary_by_method_new.csv", index=False)

    print("=" * 96)
    print("Mean over the 15 cells, ordered by mean AP")
    print("=" * 96)
    print(summ[["method", "mean_AP", "sd_AP", "mean_lift",
                "cells_above_random", "cells_best", "mean_rank"]]
          .round(4).to_string(index=False))
    print()
    print("  mean prevalence %.4f, which is what a random ordering scores"
          % cells["prevalence"].mean())
    print()

    # -------------------------------------------------- where ENS ranks
    rank_ens = cells[ap_cols].rank(axis=1, ascending=False)["AP_ENS"]
    ens = cells[["event", "asset", "n", "prevalence", "AP_ENS", "lift_ENS"]].copy()
    ens["rank_of_10"] = rank_ens.astype(int)
    ens["best_method"] = best
    ens["best_AP"] = cells[ap_cols].max(axis=1)
    ens["gap_to_best"] = ens["AP_ENS"] - ens["best_AP"]
    ens.round(6).to_csv(HERE / "pr_auc_ens_where_best_new.csv", index=False)

    print("=" * 96)
    print("Where EnsembleExpoGAF ranks in each cell")
    print("=" * 96)
    print(ens.round(4).to_string(index=False))
    print()
    won = ens[ens["rank_of_10"] == 1]
    print("  ENS is first in %d of %d cells" % (len(won), len(ens)))
    if len(won):
        for _, r in won.iterrows():
            print("     %s / %s   AP %.4f  lift over random %+.4f"
                  % (r["event"], r["asset"], r["AP_ENS"], r["lift_ENS"]))
    print()
    print("  By asset: mean AP of ENS")
    print(ens.groupby("asset", observed=True)["AP_ENS"].mean().round(4).to_string())
    print()
    print("  By episode: mean AP of ENS")
    print(ens.groupby("event", observed=True)["AP_ENS"].mean().round(4).to_string())

    # -------------------------------------------------- LaTeX table
    tex = []
    tex.append("% PR-AUC across all assets and episodes, from pr_auc_all_datasets_new.py")
    tex.append("\\begin{center}")
    tex.append("\\small")
    tex.append("\\begin{tabular}{lrrrrr}")
    tex.append("\\toprule")
    tex.append("Method & Mean PR-AUC & Mean lift & Cells above & Cells & Mean \\\\")
    tex.append("       & over 15 cells & over random & random & best & rank \\\\")
    tex.append("\\midrule")
    for _, r in summ.iterrows():
        bold = "\\textbf{%s}" if r["key"] == "ENS" else "%s"
        tex.append("%s & %.3f & %+.3f & %d & %d & %.1f \\\\"
                   % (bold % r["method"], r["mean_AP"], r["mean_lift"],
                      r["cells_above_random"], r["cells_best"], r["mean_rank"]))
    tex.append("\\midrule")
    tex.append("\\emph{Random ordering} & %.3f & 0.000 & --- & --- & --- \\\\"
               % cells["prevalence"].mean())
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{center}")
    (HERE / "pr_auc_tables_new.tex").write_text("\n".join(tex), encoding="utf-8")

    print()
    print("Wrote pr_auc_all_cells_new.csv, pr_auc_summary_by_method_new.csv,")
    print("       pr_auc_ens_where_best_new.csv, pr_auc_tables_new.tex")


if __name__ == "__main__":
    main()

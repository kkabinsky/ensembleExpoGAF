# -*- coding: utf-8 -*-
"""
standalone_tadgan_w32.py

Score the TadGAN front end alone at window length 32.

Run

    python standalone_tadgan_w32.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
OUT = os.path.join(HERE, "output")

WINDOW = 32
SIG = int(os.environ.get("TADGAN_SIGNAL_SHAPE", "100"))

ASSETS = {"USOIL": "USOIL_daily_final2.xlsx", "GOLD": "GOLD_daily_final2.xlsx",
          "EURUSD": "EURUSD_daily_final2.xlsx"}

# identical to EPISODES in the corrected runner; do not edit
EPISODES = {
    "COVID-19":        dict(out="results_covid",   data="datasets_2026",
                            train_end="2019-11-30", test_start="2019-12-01",
                            test_end="2020-06-30", onset="2020-02-20",
                            end="2020-04-20"),
    "Russia--Ukraine": dict(out="results_russia",  data="datasets_2026",
                            train_end="2021-11-30", test_start="2021-12-01",
                            test_end="2022-05-31", onset="2022-02-11",
                            end="2022-03-31"),
    "Chinese":         dict(out="results_chinese", data="datasets_2026",
                            train_end="2023-05-31", test_start="2023-06-01",
                            test_end="2023-12-29", onset="2023-08-07",
                            end="2023-09-30"),
    "Iran 2025":       dict(out="results",         data="datasets",
                            train_end="2025-03-31", test_start="2025-04-01",
                            test_end="2025-08-31", onset="2025-06-12",
                            end="2025-06-25"),
    "Iran 2026":       dict(out="results_2026",    data="datasets_2026",
                            train_end="2026-03-31", test_start="2026-04-01",
                            test_end="2026-07-09", onset="2026-06-12",
                            end="2026-06-24"),
}


def build_cell(cfg, asset):
    """Build the 32-windows, labels and split by the same rules as the
    corrected runner."""
    df = pd.read_excel(os.path.join(RUN, cfg["data"], ASSETS[asset]))
    c = {x.lower(): x for x in df.columns}
    dates = pd.to_datetime(df[c["date"]]).reset_index(drop=True)
    prices = pd.to_numeric(df[c["cp"]], errors="coerce").to_numpy(float)

    f = os.path.join(RUN, cfg["out"], asset, "tadgan_stage", "out",
                     "full_scores.csv")
    if not os.path.isfile(f):
        return None
    sc = pd.read_csv(f)["anomaly_score"].to_numpy(float)

    n_win = len(sc) - WINDOW + 1
    s_idx = np.arange(n_win)
    p_start = s_idx + SIG - 1
    p_end = p_start + WINDOW - 1
    ok = p_end < len(prices)
    s_idx, p_start, p_end = s_idx[ok], p_start[ok], p_end[ok]

    cs = int(np.where(dates >= pd.Timestamp(cfg["onset"]))[0].min())
    ce = int(np.where(dates <= pd.Timestamp(cfg["end"]))[0].max()) + 1
    y = ((p_start <= ce) & (p_end + 1 > cs)).astype(int)

    sdate = dates.iloc[p_start].reset_index(drop=True)
    te = ((sdate >= pd.Timestamp(cfg["test_start"]))
          & (sdate <= pd.Timestamp(cfg["test_end"]))).to_numpy()
    if te.sum() == 0:
        return None

    win = np.stack([sc[s:s + WINDOW] for s in s_idx])
    return dict(y=y[te], p_start=p_start[te],
                score_mean=win.mean(axis=1)[te],
                score_max=win.max(axis=1)[te])


def evaluate(y, score, order):
    if not (0 < y.sum() < len(y)):
        return None
    return dict(auc=float(roc_auc_score(y, score)),
                pr_auc=float(average_precision_score(y, score)),
                rho_score_time=float(spearmanr(score, order).statistic))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ep, cfg in EPISODES.items():
        for asset in ASSETS:
            cell = build_cell(cfg, asset)
            if cell is None:
                print("  skipped %s / %s, no TadGAN scores" % (ep, asset))
                continue
            y = cell["y"]
            order = np.arange(len(y), dtype=float)   # position in the test period
            base = dict(episode=ep, asset=asset, n=len(y), n_pos=int(y.sum()),
                        prevalence=float(y.mean()), window=WINDOW,
                        rho_label_time=float(spearmanr(y, order).statistic))
            for name, sc in (("tadgan_mean", cell["score_mean"]),
                             ("tadgan_max", cell["score_max"]),
                             ("time_index", order)):
                m = evaluate(y, sc, order)
                if m:
                    rows.append(dict(base, method=name, **m))

    t = pd.DataFrame(rows)
    order_ep = list(EPISODES)
    t["episode"] = pd.Categorical(t["episode"], order_ep, ordered=True)
    t["asset"] = pd.Categorical(t["asset"], list(ASSETS), ordered=True)
    t = t.sort_values(["episode", "asset", "method"]).reset_index(drop=True)
    t.round(6).to_csv(os.path.join(OUT, "standalone_tadgan_w32_cells.csv"),
                      index=False)

    print("=" * 100)
    print("Standalone TadGAN at window 32, same labels and split as every other arm")
    print("=" * 100)
    w = t.pivot_table(index=["episode", "asset", "n", "n_pos", "prevalence",
                             "rho_label_time"],
                      columns="method", values=["auc", "rho_score_time"],
                      observed=True)
    cols = [("auc", "tadgan_mean"), ("auc", "tadgan_max"),
            ("auc", "time_index"),
            ("rho_score_time", "tadgan_mean"), ("rho_score_time", "tadgan_max")]
    print(w[cols].round(3).to_string())

    print()
    print("=" * 100)
    print("Mean over the 15 cells")
    print("=" * 100)
    g = t.groupby("method", observed=True).agg(
        cells=("auc", "size"), mean_auc=("auc", "mean"), sd_auc=("auc", "std"),
        mean_prauc=("pr_auc", "mean"),
        mean_prev=("prevalence", "mean"),
        cells_auc_above_half=("auc", lambda v: int((v > 0.5).sum())))
    print(g.round(4).to_string())

    print()
    piv = t.pivot_table(index=["episode", "asset"], columns="method",
                        values="auc", observed=True)
    for m in ("tadgan_mean", "tadgan_max"):
        d = piv[m] - piv["time_index"]
        print("  %-12s against the time_index control: wins %d of %d cells  "
              "mean AUC difference %+.4f"
              % (m, int((d > 0).sum()), len(d), d.mean()))
    print()
    print("  How time-ordered the labels are: mean rho %.3f, min %.3f, max %.3f"
          % (t["rho_label_time"].mean(), t["rho_label_time"].min(),
             t["rho_label_time"].max()))

    g.round(6).to_csv(os.path.join(OUT, "standalone_tadgan_w32_summary.csv"))

    # ---------------------------------------------------------- LaTeX table
    tex = ["% generated by standalone_tadgan_w32.py",
           "\\begin{center}", "\\small", "\\setlength{\\tabcolsep}{4.5pt}",
           "\\begin{tabular}{llrrrrrr}", "\\toprule",
           "Episode & Asset & $n$ & Prev.\\ & TadGAN & TadGAN & Time & $\\rho$(mean \\\\",
           " & & & & mean & max & index & score, time) \\\\", "\\midrule"]
    for (ep, asset), grp in piv.groupby(level=[0, 1], observed=True):
        r = t[(t.episode == ep) & (t.asset == asset)]
        if r.empty:
            continue
        b = r.iloc[0]
        rho = float(r[r.method == "tadgan_mean"]["rho_score_time"].iloc[0])
        tex.append("%s & %s & %d & %.3f & %.3f & %.3f & %.3f & %+.3f \\\\"
                   % (ep, asset, int(b["n"]), b["prevalence"],
                      piv.loc[(ep, asset), "tadgan_mean"],
                      piv.loc[(ep, asset), "tadgan_max"],
                      piv.loc[(ep, asset), "time_index"], rho))
    tex += ["\\midrule",
            "\\multicolumn{4}{l}{\\emph{Mean AUC over 15 cells}} & %.3f & %.3f & %.3f & \\\\"
            % (g.loc["tadgan_mean", "mean_auc"], g.loc["tadgan_max", "mean_auc"],
               g.loc["time_index", "mean_auc"]),
            "\\bottomrule", "\\end{tabular}", "\\end{center}"]
    open(os.path.join(OUT, "standalone_tadgan_w32.tex"), "w",
         encoding="utf-8").write("\n".join(tex))
    print()
    print("Wrote output/standalone_tadgan_w32_cells.csv, _summary.csv, .tex")


if __name__ == "__main__":
    main()

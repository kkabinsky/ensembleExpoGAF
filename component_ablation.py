"""
component_ablation.py

Effective sample size implied by the window overlap.

Run

    python component_ablation.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "Iran_new_run")
OUT = os.path.join(HERE, "output_supplementary")

EPISODES = {"COVID-19": "results_covid", "Russia-Ukraine": "results_russia",
            "Chinese real estate": "results_chinese", "Iran 2025": "results"}
ASSETS = ["USOIL", "GOLD", "EURUSD"]
MAPPINGS = ["cosine", "arctan", "arccosh", "exponential"]
WINDOW = 32


def auc_score(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def pr_auc_score(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    npos = int(y.sum())
    if npos == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort")
    yy = y[o]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / npos)


def read(root, asset, method, mapping):
    p = os.path.join(root, asset, method, mapping, "test_scores.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p).sort_values("window_start")


def lag1(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x) == 0:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


# ---------------------------------------------------------------------------
print("=" * 92)
print("Part 1  effect of the TadGAN front end (the only thing that differs)")
print("=" * 92)

rows = []
for ep, sub in EPISODES.items():
    root = os.path.join(RUN, sub)
    if not os.path.isdir(root):
        continue
    for asset in ASSETS:
        for mp in MAPPINGS:
            h = read(root, asset, "hybrid2", mp)
            f = read(root, asset, "standalone_fanogan", mp)
            if h is None or f is None:
                continue
            y = h["label_0normal_1crash"].to_numpy(int)
            sh = pd.to_numeric(h["anomaly_score"], errors="coerce").to_numpy(float)
            sf = pd.to_numeric(f["anomaly_score"], errors="coerce").to_numpy(float)
            if len(sh) != len(sf) or not np.isfinite(sh).all():
                continue
            rows.append({
                "episode": ep, "asset": asset, "mapping": mp,
                "auc_with_tadgan": auc_score(y, sh),
                "auc_without": auc_score(y, sf),
                "prauc_with_tadgan": pr_auc_score(y, sh),
                "prauc_without": pr_auc_score(y, sf),
                "max_abs_score_diff": float(np.max(np.abs(sh - sf))),
                "score_corr": float(np.corrcoef(sh, sf)[0, 1]),
            })

t = pd.DataFrame(rows)
t["d_auc"] = (t["auc_with_tadgan"] - t["auc_without"]).round(4)
t["d_prauc"] = (t["prauc_with_tadgan"] - t["prauc_without"]).round(4)
print(t.groupby("mapping")[["auc_with_tadgan", "auc_without", "d_auc",
                            "prauc_with_tadgan", "prauc_without", "d_prauc"]]
      .mean().round(4).to_string())
print()
print("Cells in total                 : %d" % len(t))
print("Cells with AUC differing > 0.001: %d" % int((t["d_auc"].abs() > 0.001).sum()))
print("Largest AUC difference         : %.6f" % t["d_auc"].abs().max())
print("Largest raw-score difference   : %.6g" % t["max_abs_score_diff"].max())
print("Lowest correlation between arms: %.6f" % t["score_corr"].min())
t.to_csv(os.path.join(OUT, "ablation_tadgan.csv"), index=False)

# ---------------------------------------------------------------------------
print()
print("=" * 92)
print("Part 2  effect of the angular mapping")
print("=" * 92)
m = (t.groupby("mapping")
     .agg(auc=("auc_with_tadgan", "mean"), prauc=("prauc_with_tadgan", "mean"),
          n=("auc_with_tadgan", "count")).reset_index())
m = m.set_index("mapping").reindex(MAPPINGS).reset_index()
print(m.round(4).to_string(index=False))
base = float(m[m["mapping"] == "exponential"]["auc"].iloc[0])
print()
for _, r in m.iterrows():
    print("  %-12s AUC %.4f   against exponential %+0.4f"
          % (r["mapping"], r["auc"], r["auc"] - base))

# ---------------------------------------------------------------------------
print()
print("=" * 92)
print("Part 3  overlapping windows: how many independent observations")
print("=" * 92)
rows = []
for ep, sub in EPISODES.items():
    root = os.path.join(RUN, sub)
    if not os.path.isdir(root):
        continue
    for asset in ASSETS:
        d = read(root, asset, "hybrid2", "exponential")
        if d is None:
            continue
        s = pd.to_numeric(d["anomaly_score"], errors="coerce").to_numpy(float)
        y = d["label_0normal_1crash"].to_numpy(int)
        n = len(s)
        rho = lag1(s)
        n_ov = n / WINDOW
        n_ac = n * (1 - rho) / (1 + rho) if np.isfinite(rho) and rho > -1 else np.nan
        rows.append({"episode": ep, "asset": asset, "n_windows": n,
                     "n_positive": int(y.sum()), "lag1_score_corr": round(rho, 3),
                     "n_eff_overlap": round(n_ov, 1),
                     "n_eff_autocorr": round(n_ac, 1) if np.isfinite(n_ac) else None})
e = pd.DataFrame(rows)
print(e.to_string(index=False))
e.to_csv(os.path.join(OUT, "effective_sample_size.csv"), index=False)
print()
print("All cells: %d scored windows, about %.0f independent under the overlap"
      % (e["n_windows"].sum(), e["n_eff_overlap"].sum()))
print("Median lag-1 score autocorrelation: %.3f" % e["lag1_score_corr"].median())
print()
print("Output written to", OUT)

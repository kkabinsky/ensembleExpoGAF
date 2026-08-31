# -*- coding: utf-8 -*-
"""
mapping_geometry.py
===================

Measure the geometry of the four angular mappings on the scored windows.

Run from inside this folder:

    python mapping_geometry.py

Writes ``output/mapping_geometry.csv`` and prints a LaTeX table body.
"""
import os

import numpy as np
import pandas as pd

import gaf_encodings as G

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def load_windows(window=G.W_REF):
    """The raw price windows and labels, by the encoding-ablation rule."""
    series, labels = [], []
    for ep, (sub, dsub) in G.EPISODES.items():
        for asset in G.ASSETS:
            sc = os.path.join(G.RUN, sub, asset, "hybrid2", "exponential",
                              "test_scores.csv")
            px = os.path.join(G.RUN, dsub, G.ASSETS[asset])
            if not (os.path.exists(sc) and os.path.exists(px)):
                continue
            d = pd.read_csv(sc).sort_values("window_start")
            df = pd.read_excel(px)
            c = {x.lower(): x for x in df.columns}
            price = pd.to_numeric(df[c["cp"]], errors="coerce").to_numpy(float)
            st = d["window_start"].to_numpy(int)
            rng = G.crash_index_range(st,
                                      d["label_0normal_1crash"].to_numpy(int))
            if rng is None:
                continue
            cs, ce = rng
            keep = (st + window <= len(price)) & (st >= 0)
            st_w = st[keep]
            if len(st_w) == 0:
                continue
            y_w = ((st_w <= ce) & (st_w + window > cs)).astype(int)
            for s0, lab in zip(st_w, y_w):
                series.append(price[s0:s0 + window])
                labels.append(int(lab))
    return series, np.asarray(labels, int)


def fold_fraction(series, f):
    """Share of pixel pairs whose angle sum exceeds pi, over all windows."""
    over = 0
    total = 0
    for w in series:
        phi = f(G.minmax01(G.resize_1d(np.asarray(w, float), G.IMG)))
        ssum = phi[:, None] + phi[None, :]
        over += int(np.count_nonzero(ssum > np.pi))
        total += ssum.size
    return over / total if total else float("nan")


def fisher(x, y):
    """Between-class mean distance, pooled within-class spread, and the ratio."""
    a, b = x[y == 1], x[y == 0]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    between = float(np.linalg.norm(ma - mb))
    wa = float(np.sqrt(np.mean(np.sum((a - ma) ** 2, axis=1))))
    wb = float(np.sqrt(np.mean(np.sum((b - mb) ** 2, axis=1))))
    within = float(np.sqrt(0.5 * (wa ** 2 + wb ** 2)))
    return between, within, (between / within if within > 0 else np.nan)


def main():
    series, y = load_windows()
    data = G.build()
    grid = np.linspace(0.0, 1.0, G.IMG)
    rows = []

    for m in G.ORDER:
        f = G.MAPS[m]
        phi_grid = f(grid)
        d = np.gradient(phi_grid, grid)

        flat = np.asarray(data[m]["X"], dtype=np.float64)
        flat = flat.reshape(len(flat), -1)
        yy = np.asarray(data[m]["y"], int)
        between, within, fr = fisher(flat, yy)

        rows.append(dict(
            mapping=m,
            phi_max=float(phi_grid.max()),
            pair_sum_max=float(2.0 * phi_grid.max()),
            can_fold=bool(2.0 * phi_grid.max() > np.pi + 1e-12),
            fold_fraction=fold_fraction(series, f),
            dphi_low=float(d[1]), dphi_high=float(d[-2]),
            resolution_rises=bool(d[-2] > d[1]),
            between=between, within=within, fisher_ratio=fr,
            n_windows=int(len(yy)), n_event=int(yy.sum())))

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "mapping_geometry.csv"), index=False)

    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda v: "%.4f" % v))

    print()
    print("LaTeX table body:")
    NAME = {"cosine": "Cosine", "arctan": "Arctan",
            "arccosh": "Arccosh", "exponential": "Exponential"}
    for _, r in df.iterrows():
        print("%-12s & $[0,\\,%.3f]$ & %s & $%.1f$ & $%.2f$ & $%.2f$ & "
              "$%.2f$ & $%.2f$ & $%.2f$ \\\\"
              % (NAME[r["mapping"]], r["phi_max"],
                 "yes" if r["can_fold"] else "no",
                 100.0 * r["fold_fraction"],
                 r["dphi_low"], r["dphi_high"],
                 r["between"], r["within"], r["fisher_ratio"]))


if __name__ == "__main__":
    main()

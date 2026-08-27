# -*- coding: utf-8 -*-
"""
cnn_window_sweep.py
===================

Does the window length change the ordering of the mappings? Answers Reviewer 4
point 9.

Design
    The image is 32x32 at every window length, because the mapping resamples the
    window to the image size before the angular transform. The network and its
    parameter count are therefore identical throughout and only the span of
    history summarised by one image changes.

    The set of window start positions is also identical at every length, so the
    test dates coincide and the comparison is paired cell by cell.

Three window lengths by four mappings by four held-out episodes by three seeds,
under leave-one-episode-out: windows inside an episode overlap by 31 of their 32
observations, so a random split would leak almost completely.

What cannot be held constant
    A window is labelled positive when it overlaps the event interval, so a
    longer window is labelled positive more often and the effective sample size
    falls as the window lengthens. AUC is therefore not comparable across window
    lengths; only the ordering within a length is. Both the positive rate and
    the effective sample size are reported beside the AUC.

Run
    python cnn_window_sweep.py
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "Iran_new_run")
OUT = os.path.join(HERE, "output")

EPISODES = {"COVID-19": ("results_covid", "datasets"),
            "Russia-Ukraine": ("results_russia", "datasets"),
            "Chinese": ("results_chinese", "datasets"),
            "Iran 2025": ("results", "datasets")}
ASSETS = {"USOIL": "USOIL_daily_final2.xlsx", "GOLD": "GOLD_daily_final2.xlsx",
          "EURUSD": "EURUSD_daily_final2.xlsx"}
W_REF = 32          # the window the manuscript uses, and the labels refer to
IMG = 32            # image size, fixed, so the network never changes
ORDER = ["cosine", "arctan", "arccosh", "exponential"]
MAPS = {
    "cosine":      lambda s: np.arccos(np.clip(2 * s - 1, -1, 1)),
    "arctan":      lambda s: np.arctan(s),
    "arccosh":     lambda s: np.arccosh(1.0 + s),
    "exponential": lambda s: np.pi * (np.exp(s) - 1.0) / (np.e - 1.0),
}


def minmax01(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def resize_1d(x, n):
    """Resample a series to the image size, as the f-AnoGAN stage does."""
    if len(x) == n:
        return x
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def gaf_image(series, f):
    phi = f(minmax01(resize_1d(np.asarray(series, float), IMG)))
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def auc_score(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def crash_index_range(starts, labels):
    """Recover the event interval from the window-32 labels."""
    pos = starts[labels == 1]
    if len(pos) == 0:
        return None
    return int(pos.min()) + W_REF - 1, int(pos.max())


def build(w, w_max):
    """Build the images at window length w, with the same start positions at

    every length. The cut uses w_max rather than w so that all lengths share
    exactly the same starts; cutting by w would drop trailing rows at the
    longer lengths and the paired comparison would no longer line up.
    """
    data = {m: {"X": [], "y": [], "ep": []} for m in ORDER}
    for ep, (sub, dsub) in EPISODES.items():
        for asset in ASSETS:
            sc = os.path.join(RUN, sub, asset, "hybrid2", "exponential",
                              "test_scores.csv")
            px = os.path.join(RUN, dsub, ASSETS[asset])
            if not (os.path.exists(sc) and os.path.exists(px)):
                continue
            d = pd.read_csv(sc).sort_values("window_start")
            df = pd.read_excel(px)
            c = {x.lower(): x for x in df.columns}
            price = pd.to_numeric(df[c["cp"]], errors="coerce").to_numpy(float)
            st = d["window_start"].to_numpy(int)
            y32 = d["label_0normal_1crash"].to_numpy(int)
            rng = crash_index_range(st, y32)
            if rng is None:
                continue
            cs, ce = rng
            # cut by the longest window in the sweep so all lengths align
            keep = (st + w_max <= len(price)) & (st >= 0)
            st_w = st[keep]
            if len(st_w) == 0:
                continue
            y_w = ((st_w <= ce) & (st_w + w > cs)).astype(int)
            for m in ORDER:
                f = MAPS[m]
                for s, lab in zip(st_w, y_w):
                    data[m]["X"].append(gaf_image(price[s:s + w], f))
                    data[m]["y"].append(int(lab))
                    data[m]["ep"].append(ep)
    for m in ORDER:
        if not data[m]["X"]:
            return None
        data[m]["X"] = np.stack(data[m]["X"])[:, None, :, :]
        data[m]["y"] = np.asarray(data[m]["y"], int)
        data[m]["ep"] = np.asarray(data[m]["ep"], object)
    return data


def make_model(kernel=3, ch1=8, ch2=16, dropout=0.3):
    p = kernel // 2
    return nn.Sequential(
        nn.Conv2d(1, ch1, kernel, padding=p), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(ch1, ch2, kernel, padding=p), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(dropout), nn.Linear(ch2, 1), nn.Flatten(0))


def train_eval(model, Xtr, ytr, Xte, epochs, seed, lr=1e-3):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = float(ytr.sum())
    w = torch.tensor(max(len(ytr) - pos, 1.0) / max(pos, 1.0),
                     dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    n, bs = len(xt), 64
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xte, dtype=torch.float32)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="16,32,64")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    ws = [int(x) for x in a.windows.split(",")]

    csv_path = os.path.join(OUT, "cnn_window_sweep%s.csv" % a.tag)
    rows, seen = [], set()
    if os.path.exists(csv_path):
        try:
            prev = pd.read_csv(csv_path)
            rows = prev.to_dict("records")
            seen = {(int(r["window"]), int(r.get("w_max", max(ws))),
                     str(r["mapping"]), str(r["held_out"]), int(r["seed"]))
                    for _, r in prev.iterrows()}
            print("Found %d earlier rows; resuming" % len(seen), flush=True)
        except Exception as e:
            print("Could not read earlier results (%s); starting over" % e, flush=True)

    def flush():
        tmp = csv_path + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, csv_path)

    t0 = time.time()
    w_max = max(ws)
    for w in ws:
        data = build(w, w_max)
        if data is None:
            print("window %d  skipped, images could not be built" % w, flush=True)
            continue
        y_all = data[ORDER[0]]["y"]
        print("window %2d   %d images per mapping   positive rate %.3f   "
              "image %dx%d" % (w, len(y_all), y_all.mean(), IMG, IMG),
              flush=True)
        for m in ORDER:
            X, y, ep = data[m]["X"], data[m]["y"], data[m]["ep"]
            for held in EPISODES:
                te = ep == held
                tr = ~te
                if te.sum() == 0 or y[te].sum() in (0, int(te.sum())):
                    continue
                for sd in range(a.seeds):
                    if (w, w_max, m, held, sd) in seen:
                        continue
                    torch.manual_seed(sd)
                    mdl = make_model()
                    sc = train_eval(mdl, X[tr], y[tr], X[te], a.epochs, sd)
                    rows.append({"window": w, "w_max": w_max, "mapping": m,
                                 "held_out": held,
                                 "seed": sd, "auc": auc_score(y[te], sc),
                                 "n": int(te.sum()),
                                 "prevalence": float(y[te].mean()),
                                 "n_eff": int(te.sum()) / float(w),
                                 "img_size": IMG, "epochs": a.epochs})
            flush()
        print("    finished window %d   %.0f min elapsed"
              % (w, (time.time() - t0) / 60), flush=True)

    report(csv_path)


def report(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["auc"].notna()]
    if df.empty:
        print("no results yet")
        return
    ws = sorted(df["window"].unique())
    print()
    print("=" * 92)
    print("Mean AUC over episodes and seeds; rows are window length, columns mapping")
    print("=" * 92)
    p = df.pivot_table(index="window", columns="mapping", values="auc",
                       aggfunc="mean")
    p = p[[c for c in ORDER if c in p.columns]]
    p["best"] = p[[c for c in ORDER if c in p.columns]].idxmax(axis=1)
    ctx = df.groupby("window").agg(positive=("prevalence", "mean"),
                                   n_eff=("n_eff", "mean"))
    print(p.join(ctx).round(4).to_string())

    print()
    print("=" * 92)
    print("Rank of exponential at each window length; 1 = best of the four")
    print("=" * 92)
    n_first = 0
    for w in ws:
        r = p.loc[w, [c for c in ORDER if c in p.columns]].astype(float)
        rank = int(r.rank(ascending=False)["exponential"])
        n_first += int(rank == 1)
        print("  window %-3d  exponential AUC %.4f  rank %d   "
              "best is %s %.4f"
              % (w, r["exponential"], rank, r.idxmax(), r.max()))
    print()
    print("  exponential is first at %d of %d window lengths" % (n_first, len(ws)))

    # paired within a cell; episode and seed must match
    print()
    print("=" * 92)
    print("Paired against window 32, same cell and same seed")
    print("=" * 92)
    piv = df.pivot_table(index=["mapping", "held_out", "seed"],
                         columns="window", values="auc")
    if W_REF in piv.columns:
        from scipy.stats import binomtest
        print("%-14s %6s %9s %9s %8s" % ("window", "cells", "mean diff",
                                          "wins", "p"))
        for w in [c for c in piv.columns if c != W_REF]:
            d = (piv[w] - piv[W_REF]).dropna()
            if d.empty:
                continue
            wins = int((d > 0).sum())
            print("%-14d %6d %9.4f %5d/%-3d %8.3f"
                  % (w, len(d), d.mean(), wins, len(d),
                     binomtest(wins, len(d), 0.5).pvalue))
    print()
    print("How to read this")
    print("  AUC cannot be compared directly across window lengths")
    print("  because the overlap labelling rule changes the positive rate")
    print("  what is comparable is the ordering of mappings within a length")
    print("  if the ordering holds at every length, it does not rest on 32")
    print()
    print("Output file", csv_path)


if __name__ == "__main__":
    main()

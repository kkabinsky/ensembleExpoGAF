# -*- coding: utf-8 -*-
"""
seed_variability.py
===================

Repeat the trained stage at several seeds and report the spread.

Run
    python seed_variability.py --seeds 5
    python seed_variability.py --seeds 5 --episodes covid,russia
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "Iran_new_run")
OUT = os.path.join(HERE, "output")

os.environ.setdefault("ALLOW_CPU", "1")
sys.path.insert(0, RUN)
import fanogan_four_gaf_compare as fa                            # noqa: E402
from torch.utils.data import DataLoader, TensorDataset           # noqa: E402

if not torch.cuda.is_available():
    fa.device = torch.device("cpu")

ASSETS = {"USOIL": "USOIL_daily_final2.xlsx",
          "GOLD": "GOLD_daily_final2.xlsx",
          "EURUSD": "EURUSD_daily_final2.xlsx"}

# the same dates the pipeline was run with, so the cells line up with the paper
EPISODES = {
    "covid":    dict(data="datasets_2026", train_end="2019-11-30",
                     test_start="2019-12-01", test_end="2020-06-30",
                     onset="2020-02-20", end="2020-04-20"),
    "russia":   dict(data="datasets_2026", train_end="2021-11-30",
                     test_start="2021-12-01", test_end="2022-05-31",
                     onset="2022-02-11", end="2022-03-31"),
    "chinese":  dict(data="datasets_2026", train_end="2023-05-31",
                     test_start="2023-06-01", test_end="2023-12-29",
                     onset="2023-08-07", end="2023-09-30"),
    "iran2025": dict(data="datasets", train_end="2025-03-31",
                     test_start="2025-04-01", test_end="2025-08-31",
                     onset="2025-06-12", end="2025-06-25"),
    "iran2026": dict(data="datasets_2026", train_end="2026-03-31",
                     test_start="2026-04-01", test_end="2026-07-09",
                     onset="2026-06-12", end="2026-06-24"),
}


def build(cfg, asset, mapping, window):
    df = pd.read_excel(os.path.join(RUN, cfg["data"], ASSETS[asset]))
    c = {x.lower(): x for x in df.columns}
    dates = pd.to_datetime(df[c["date"]]).reset_index(drop=True)
    price = pd.to_numeric(df[c["cp"]], errors="coerce").to_numpy(float)
    starts = np.arange(len(price) - window + 1)
    hit = np.flatnonzero(dates >= pd.Timestamp(cfg["onset"]))
    if len(hit) == 0:
        return None
    cs = int(hit.min())
    ce = int(np.flatnonzero(dates <= pd.Timestamp(cfg["end"])).max()) + 1
    y = ((starts <= ce) & (starts + window > cs)).astype(int)
    sd = dates.iloc[starts].reset_index(drop=True)
    tr = (sd <= pd.Timestamp(cfg["train_end"])).to_numpy()
    te = ((sd >= pd.Timestamp(cfg["test_start"]))
          & (sd <= pd.Timestamp(cfg["test_end"]))).to_numpy()
    gaf = fa.GAF_METHODS[mapping]
    imgs = np.stack([gaf(price[s:s + window], fa.IMG_SIZE) for s in starts])
    return imgs, y, tr & (y == 0), te


def fit_score(x_tr, x_te, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fa.N_EPOCHS_GAN = epochs
    fa.N_EPOCHS_ENC = epochs
    loader = DataLoader(
        TensorDataset(torch.tensor(x_tr, dtype=torch.float32).unsqueeze(1)),
        batch_size=fa.BATCH_SIZE, shuffle=True, drop_last=True)
    G = fa.Generator().to(fa.device)
    D = fa.Discriminator().to(fa.device)
    E = fa.Encoder().to(fa.device)
    fa.train_wgangp(G, D, loader)
    fa.train_encoder(E, G, D, loader)
    return fa.anomaly_scores(E, G, D, x_te)


def auc(y, s):
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def average_precision(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    yy = y[o]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", default="covid,russia,chinese,iran2025,iran2026")
    ap.add_argument("--assets", default="USOIL,GOLD,EURUSD")
    ap.add_argument("--mapping", default="exponential")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    eps = [x.strip() for x in a.episodes.split(",") if x.strip()]
    assets = [x.strip() for x in a.assets.split(",") if x.strip()]

    csv = os.path.join(OUT, "seed_variability%s.csv" % a.tag)
    rows, done = [], set()
    if os.path.exists(csv):
        try:
            prev = pd.read_csv(csv)
            rows = prev.to_dict("records")
            done = {(str(r["episode"]), str(r["asset"]), int(r["seed"]))
                    for _, r in prev.iterrows()}
            print("found %d finished cells; continuing from there" % len(done),
                  flush=True)
        except Exception as e:
            print("could not read earlier results (%s); starting over" % e,
                  flush=True)

    def flush():
        tmp = csv + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, csv)

    total = len(eps) * len(assets) * a.seeds
    print("device %s   mapping %s   window %d   epochs %d   seeds %d"
          % (fa.device, a.mapping, a.window, a.epochs, a.seeds), flush=True)
    print("%d cells to do in total" % total, flush=True)
    print(flush=True)

    t0 = time.time()
    n = 0
    for ep in eps:
        cfg = EPISODES[ep]
        for asset in assets:
            built = build(cfg, asset, a.mapping, a.window)
            if built is None:
                print("%-9s %-7s skipped, the episode is not in this vintage"
                      % (ep, asset), flush=True)
                continue
            imgs, y, trn, te = built
            if trn.sum() < max(fa.BATCH_SIZE, 8) or te.sum() == 0:
                print("%-9s %-7s skipped, not enough data" % (ep, asset),
                      flush=True)
                continue
            yt = y[te]
            for sd in range(a.seeds):
                n += 1
                if (ep, asset, sd) in done:
                    continue
                s = fit_score(imgs[trn], imgs[te], a.epochs, sd)
                ok = 0 < yt.sum() < len(yt)
                rows.append({
                    "episode": ep, "asset": asset, "seed": sd,
                    "mapping": a.mapping, "window": a.window,
                    "auc": auc(yt, s) if ok else float("nan"),
                    "prauc": average_precision(yt, s) if ok else float("nan"),
                    "n_train": int(trn.sum()), "n_test": int(te.sum()),
                    "prevalence": float(yt.mean()),
                    "n_eff": int(te.sum()) / float(a.window),
                    "epochs": a.epochs})
                flush()
                print("%-9s %-7s seed %d   AUC %.4f   [%d/%d  %.0f min]"
                      % (ep, asset, sd, rows[-1]["auc"], n, total,
                         (time.time() - t0) / 60), flush=True)

    report(csv)


def report(csv):
    r = pd.read_csv(csv)
    r = r[r["auc"].notna()]
    if r.empty:
        print("no results yet")
        return
    print()
    print("=" * 80)
    print("Spread across seeds within each cell")
    print("=" * 80)
    g = r.groupby(["episode", "asset"]).agg(
        seeds=("auc", "size"), mean=("auc", "mean"), sd=("auc", "std"),
        low=("auc", "min"), high=("auc", "max"))
    g["range"] = g["high"] - g["low"]
    print(g.round(4).to_string())

    print()
    print("=" * 80)
    print("What this means for every other comparison in the paper")
    print("=" * 80)
    sd = float(g["sd"].mean())
    rng = float(g["range"].mean())
    print("  mean standard deviation across seeds within a cell   %.4f" % sd)
    print("  mean range from the worst seed to the best           %.4f" % rng)
    print("  mean AUC over all cells and seeds                    %.4f"
          % float(r["auc"].mean()))
    print()
    print("  A difference between two methods that is smaller than %.3f is")
    print("  within what a change of seed produces on its own." % rng)

    print()
    print("=" * 80)
    print("Would the conclusion change if a different seed had been used?")
    print("=" * 80)
    piv = r.pivot_table(index=["episode", "asset"], columns="seed",
                        values="auc")
    best = piv.mean(axis=0).idxmax()
    worst = piv.mean(axis=0).idxmin()
    print("  best seed overall  %d, mean AUC %.4f"
          % (best, piv[best].mean()))
    print("  worst seed overall %d, mean AUC %.4f"
          % (worst, piv[worst].mean()))
    print("  the gap between reporting the best and the worst seed is %.4f"
          % (piv[best].mean() - piv[worst].mean()))
    print()
    print("Output file", csv)


if __name__ == "__main__":
    main()

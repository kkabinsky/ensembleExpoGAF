# -*- coding: utf-8 -*-
"""
window_size_check.py
====================

Does the window length change the result? Answers Reviewer 4 point 9 on the
one-class scorer.

The image size is held at 32x32 for every window length, so the network and its
parameter count are identical throughout and only the span of history summarised
by one image changes. The set of window start positions is also held fixed, so
the test dates coincide and the comparison is paired cell by cell.

AUC is not comparable across window lengths: a window is labelled positive when
it overlaps the event interval, so a longer window is labelled positive more
often and the effective sample size falls as the window lengthens. Both are
reported beside the AUC and must be read with it.

Run
    python window_size_check.py
    python window_size_check.py --threads 4
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ALLOW_CPU", "1")
import fanogan_four_gaf_compare as fa
if not torch.cuda.is_available():
    fa.device = torch.device("cpu")
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = os.path.join(os.path.dirname(HERE), "output_window_check")

ASSETS = {"USOIL": "USOIL_daily_final2.xlsx", "GOLD": "GOLD_daily_final2.xlsx",
          "EURUSD": "EURUSD_daily_final2.xlsx"}

# identical settings to the corrected runner so the two are comparable
EPISODES = {
    "covid":   dict(data="datasets_2026", train_end="2019-11-30",
                    test_start="2019-12-01", test_end="2020-06-30",
                    onset="2020-02-20", end="2020-04-20"),
    "russia":  dict(data="datasets_2026", train_end="2021-11-30",
                    test_start="2021-12-01", test_end="2022-05-31",
                    onset="2022-02-11", end="2022-03-31"),
    "chinese": dict(data="datasets_2026", train_end="2023-05-31",
                    test_start="2023-06-01", test_end="2023-12-29",
                    onset="2023-08-07", end="2023-09-30"),
    "iran2025": dict(data="datasets", train_end="2025-03-31",
                     test_start="2025-04-01", test_end="2025-08-31",
                     onset="2025-06-12", end="2025-06-25"),
    "iran2026": dict(data="datasets_2026", train_end="2026-03-31",
                     test_start="2026-04-01", test_end="2026-07-09",
                     onset="2026-06-12", end="2026-06-24"),
}


def build(cfg, asset, mapping, w):
    """Build GAF images at window length w; the image size never changes."""
    df = pd.read_excel(os.path.join(HERE, cfg["data"], ASSETS[asset]))
    c = {x.lower(): x for x in df.columns}
    dates = pd.to_datetime(df[c["date"]]).reset_index(drop=True)
    price = pd.to_numeric(df[c["cp"]], errors="coerce").to_numpy(float)

    starts = np.arange(len(price) - w + 1)
    hit = np.where(dates >= pd.Timestamp(cfg["onset"]))[0]
    if len(hit) == 0:
        return None
    cs = int(hit.min())
    ce = int(np.where(dates <= pd.Timestamp(cfg["end"]))[0].max()) + 1
    y = ((starts <= ce) & (starts + w > cs)).astype(int)

    sd = dates.iloc[starts].reset_index(drop=True)
    tr = (sd <= pd.Timestamp(cfg["train_end"])).to_numpy()
    te = ((sd >= pd.Timestamp(cfg["test_start"]))
          & (sd <= pd.Timestamp(cfg["test_end"]))).to_numpy()

    gaf = fa.GAF_METHODS[mapping]
    imgs = np.stack([gaf(price[s:s + w], fa.IMG_SIZE) for s in starts])
    return imgs, y, tr & (y == 0), te


def fit_score(x_tr, x_te, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fa.N_EPOCHS_GAN = epochs
    fa.N_EPOCHS_ENC = epochs
    loader = DataLoader(
        TensorDataset(torch.tensor(x_tr, dtype=torch.float32).unsqueeze(1)),
        batch_size=fa.BATCH_SIZE, shuffle=True, drop_last=True)
    G, D, E = (fa.Generator().to(fa.device), fa.Discriminator().to(fa.device),
               fa.Encoder().to(fa.device))
    fa.train_wgangp(G, D, loader)
    fa.train_encoder(E, G, D, loader)
    return fa.anomaly_scores(E, G, D, x_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="16,32,64")
    ap.add_argument("--episodes", default="covid,russia,iran2025")
    ap.add_argument("--assets", default="USOIL,GOLD,EURUSD")
    ap.add_argument("--mapping", default="exponential")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0,
                    help="cap torch threads when sharing the machine")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)

    ws = [int(x) for x in a.windows.split(",")]
    eps = [e.strip() for e in a.episodes.split(",") if e.strip()]
    assets = [x.strip() for x in a.assets.split(",") if x.strip()]

    csv = os.path.join(OUT, "window_check_%s.csv" % a.mapping)
    rows, done = [], set()
    if os.path.exists(csv):
        try:
            prev = pd.read_csv(csv)
            rows = prev.to_dict("records")
            done = {(str(r["episode"]), str(r["asset"]), int(r["window"]),
                     int(r["seed"])) for _, r in prev.iterrows()}
            print("Found %d earlier rows; resuming from there" % len(done), flush=True)
        except Exception as e:
            print("Could not read earlier results (%s); starting over" % e, flush=True)

    def flush():
        tmp = csv + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, csv)

    total = len(eps) * len(assets) * len(ws) * a.seeds
    print("Device %s   image %dx%d fixed throughout   %d cells to run"
          % (fa.device, fa.IMG_SIZE, fa.IMG_SIZE, total), flush=True)
    print("epoch %d  batch %d  n_critic %d"
          % (a.epochs, fa.BATCH_SIZE, fa.N_CRITIC), flush=True)
    print(flush=True)

    t0 = time.time()
    n = 0
    for ep in eps:
        cfg = EPISODES[ep]
        for asset in assets:
            for w in ws:
                built = build(cfg, asset, a.mapping, w)
                if built is None:
                    print("%s %s window %d  skipped, no event interval in the data"
                          % (ep, asset, w), flush=True)
                    continue
                imgs, y, trn, te = built
                if trn.sum() < max(fa.BATCH_SIZE, 8) or te.sum() == 0:
                    print("%s %s window %d  skipped, not enough data" % (ep, asset, w),
                          flush=True)
                    continue
                yt = y[te]
                n_eff = int(te.sum()) / float(w)
                print("%-9s %-7s window %2d   train %3d  test %3d  "
                      "positive %.3f  effective n about %.1f"
                      % (ep, asset, w, int(trn.sum()), int(te.sum()),
                         yt.mean(), n_eff), flush=True)
                for sd in range(a.seeds):
                    n += 1
                    if (ep, asset, w, sd) in done:
                        print("    seed %d  skipped, already done" % sd, flush=True)
                        continue
                    sc = fit_score(imgs[trn], imgs[te], a.epochs, sd)
                    ok = 0 < yt.sum() < len(yt)
                    auc = float(roc_auc_score(yt, sc)) if ok else float("nan")
                    pra = (float(average_precision_score(yt, sc)) if ok
                           else float("nan"))
                    rows.append({"episode": ep, "asset": asset, "window": w,
                                 "seed": sd, "mapping": a.mapping,
                                 "auc": auc, "prauc": pra,
                                 "n_train": int(trn.sum()),
                                 "n_test": int(te.sum()),
                                 "prevalence": float(yt.mean()),
                                 "n_eff": n_eff, "epochs": a.epochs,
                                 "img_size": fa.IMG_SIZE})
                    flush()
                    print("    seed %d -> AUC %.4f  PR-AUC %.4f   "
                          "[%d/%d  %.0f min]"
                          % (sd, auc, pra, n, total, (time.time() - t0) / 60),
                          flush=True)

    summarise(csv)


def summarise(csv):
    r = pd.read_csv(csv)
    r = r[r["auc"].notna()]
    if r.empty:
        print("no results yet")
        return
    print()
    print("=" * 78)
    print("Summary: window length against result, mapping %s" % r["mapping"].iloc[0])
    print("=" * 78)
    g = r.groupby("window").agg(
        n_cell=("auc", "size"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
        prauc_mean=("prauc", "mean"), prevalence=("prevalence", "mean"),
        n_eff=("n_eff", "mean"))
    print(g.round(4).to_string())

    # paired within a cell; every window length against the 32 the paper uses
    piv = r.pivot_table(index=["episode", "asset", "seed"], columns="window",
                        values="auc")
    if 32 in piv.columns:
        print()
        print("Paired against window 32, the length the manuscript uses")
        print("%-8s %8s %8s %8s %8s" % ("window", "cells", "diff", "wins", "p"))
        from scipy.stats import binomtest
        for w in sorted(c for c in piv.columns if c != 32):
            d = (piv[w] - piv[32]).dropna()
            if d.empty:
                continue
            wins = int((d > 0).sum())
            p = binomtest(wins, len(d), 0.5).pvalue
            print("%-8d %8d %8.4f %5d/%-3d %7.3f"
                  % (w, len(d), d.mean(), wins, len(d), p))
    print()
    print("How to read this")
    print("  the image size is fixed, so the network is identical throughout")
    print("  read n_eff beside the AUC; shorter windows give more of them")
    print("  the positive rate also differs, because the label rule is overlap")
    print()
    print("Output file", csv)


if __name__ == "__main__":
    main()

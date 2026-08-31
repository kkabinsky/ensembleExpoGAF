# -*- coding: utf-8 -*-
"""
run_hybrid_true.py

Date-based runner: the image is built from the TadGAN score window.

Reads
    <OUT_DIR>/<asset>/hybrid_true/<mapping>/test_scores.csv

Run

    python run_hybrid_true.py
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
import fanogan_four_gaf_compare as fa
if not torch.cuda.is_available():
    fa.device = torch.device("cpu")
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score

WINDOW = fa.WINDOW_SIZE
SIG = int(os.environ.get("TADGAN_SIGNAL_SHAPE", "100"))
MAPS = ["exponential", "cosine", "arccosh", "arctan"]
ASSETS = {"USOIL": "USOIL_daily_final2.xlsx", "GOLD": "GOLD_daily_final2.xlsx",
          "EURUSD": "EURUSD_daily_final2.xlsx"}

# every episode; the values match the original .cmd files exactly
EPISODES = {
    "covid":   dict(out="results_covid",   data="datasets_2026",
                    train_end="2019-11-30", test_start="2019-12-01",
                    test_end="2020-06-30", onset="2020-02-20", end="2020-04-20"),
    "russia":  dict(out="results_russia",  data="datasets_2026",
                    train_end="2021-11-30", test_start="2021-12-01",
                    test_end="2022-05-31", onset="2022-02-11", end="2022-03-31"),
    "chinese": dict(out="results_chinese", data="datasets_2026",
                    train_end="2023-05-31", test_start="2023-06-01",
                    test_end="2023-12-29", onset="2023-08-07", end="2023-09-30"),
    "iran2025": dict(out="results",        data="datasets",
                    train_end="2025-03-31", test_start="2025-04-01",
                    test_end="2025-08-31", onset="2025-06-12", end="2025-06-25"),
    "iran2026": dict(out="results_2026",   data="datasets_2026",
                    train_end="2026-03-31", test_start="2026-04-01",
                    test_end="2026-07-09", onset="2026-06-12", end="2026-06-24"),
    # the crash-free control, used to set thresholds and measure false alarms
    "control": dict(out=os.path.join("covid_normal", "results"), data="datasets",
                    train_end="2019-06-30", test_start="2019-07-01",
                    test_end="2019-12-31", onset="2020-02-20", end="2020-04-20"),
}


def load_prices(data_subdir, asset):
    df = pd.read_excel(os.path.join(HERE, data_subdir, ASSETS[asset]))
    c = {x.lower(): x for x in df.columns}
    d = pd.to_datetime(df[c["date"]]).reset_index(drop=True)
    p = pd.to_numeric(df[c["cp"]], errors="coerce").to_numpy(float)
    return p, d


def load_tadgan_scores(out_dir, asset):
    """The cached TadGAN scores; nothing is retrained."""
    f = os.path.join(HERE, out_dir, asset, "tadgan_stage", "out", "full_scores.csv")
    if not os.path.isfile(f):
        return None
    d = pd.read_csv(f)
    if "anomaly_score" not in d.columns:
        return None
    return d["anomaly_score"].to_numpy(float)


def fanogan_scores(x_train, x_test):
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)),
        batch_size=fa.BATCH_SIZE, shuffle=True, drop_last=True)
    G = fa.Generator().to(fa.device)
    D = fa.Discriminator().to(fa.device)
    E = fa.Encoder().to(fa.device)
    fa.train_wgangp(G, D, loader)
    fa.train_encoder(E, G, D, loader)
    return (fa.anomaly_scores(E, G, D, x_train),
            fa.anomaly_scores(E, G, D, x_test))


def run_cell(ep_name, cfg, asset, mapping, force=False):
    both = [os.path.join(HERE, cfg["out"], asset, arm, mapping, "test_scores.csv")
            for arm in ("hybrid_true", "standalone_ref")]
    if all(os.path.isfile(f) for f in both) and not force:
        return "skipped, both arms already done"

    prices, dates = load_prices(cfg["data"], asset)
    sc = load_tadgan_scores(cfg["out"], asset)
    if sc is None:
        return "no TadGAN scores"
    # guard against an unexpected length
    if len(sc) < WINDOW + 5:
        return "score series too short"

    # score window s -> price range [s+SIG-1, s+SIG-1+WINDOW-1]
    n_win = len(sc) - WINDOW + 1
    s_idx = np.arange(n_win)
    p_start = s_idx + SIG - 1
    p_end = p_start + WINDOW - 1
    ok = p_end < len(prices)
    s_idx, p_start, p_end = s_idx[ok], p_start[ok], p_end[ok]
    if len(s_idx) == 0:
        return "index alignment failed"

    # labels by the same rule as the submitted runner: a window overlapping
    # the event interval is positive
    cs = int(np.where(dates >= pd.Timestamp(cfg["onset"]))[0].min()) \
        if (dates >= pd.Timestamp(cfg["onset"])).any() else len(prices) + 1
    ce_arr = np.where(dates <= pd.Timestamp(cfg["end"]))[0]
    ce = int(ce_arr.max()) + 1 if len(ce_arr) else cs
    y = ((p_start <= ce) & (p_end + 1 > cs)).astype(int)

    sdate = dates.iloc[p_start].reset_index(drop=True)
    tr = (sdate <= pd.Timestamp(cfg["train_end"])).to_numpy()
    te = ((sdate >= pd.Timestamp(cfg["test_start"]))
          & (sdate <= pd.Timestamp(cfg["test_end"]))).to_numpy()
    trn = tr & (y == 0)
    if trn.sum() < max(fa.BATCH_SIZE, 8) or te.sum() == 0:
        return "not enough data (train=%d test=%d)" % (int(trn.sum()), int(te.sum()))

    gaf_fn = fa.GAF_METHODS[mapping]
    yt = y[te]

    # -----------------------------------------------------------------
    # train both arms with identical settings; only the input to the GAF differs
    #   hybrid_true    GAF of TadGAN scores  <- the architecture as described
    #   standalone_ref GAF of raw prices     <- the comparison arm
    # both must be trained in the same round; do not compare against older runs
    # which used a different epoch budget and would confound two factors
    # -----------------------------------------------------------------
    inputs = {
        "hybrid_true": np.stack([gaf_fn(sc[s:s + WINDOW], fa.IMG_SIZE)
                                 for s in s_idx]),
        "standalone_ref": np.stack([gaf_fn(prices[a:a + WINDOW], fa.IMG_SIZE)
                                    for a in p_start]),
    }

    msg = []
    for arm, imgs in inputs.items():
        d = os.path.join(HERE, cfg["out"], asset, arm, mapping)
        if os.path.isfile(os.path.join(d, "test_scores.csv")) and not force:
            msg.append("%s skipped" % arm)
            continue
        torch.manual_seed(fa.RANDOM_SEED)
        np.random.seed(fa.RANDOM_SEED)
        _, tes = fanogan_scores(imgs[trn], imgs[te])
        os.makedirs(d, exist_ok=True)
        thr = float(np.quantile(tes, fa.THRESHOLD_Q))
        pred = (tes > thr).astype(int)
        pd.DataFrame({"window_start": p_start[te], "window_end": p_end[te],
                      "label_0normal_1crash": yt, "anomaly_score": tes,
                      "predicted_label": pred}).to_csv(
            os.path.join(d, "test_scores.csv"), index=False)
        try:
            auc = (float(roc_auc_score(yt, tes))
                   if 0 < yt.sum() < len(yt) else float("nan"))
        except Exception:
            auc = float("nan")
        pd.DataFrame([{"dataset": asset, "method": arm, "mapping": mapping,
                       "auc": auc, "f1": float(f1_score(yt, pred, zero_division=0)),
                       "n_test": int(len(yt)), "n_crash": int(yt.sum()),
                       "n_epochs_gan": fa.N_EPOCHS_GAN,
                       "n_epochs_enc": fa.N_EPOCHS_ENC,
                       "batch_size": fa.BATCH_SIZE,
                       "n_critic": fa.N_CRITIC}]).to_csv(
            os.path.join(d, "metrics_summary.csv"), index=False)
        msg.append("%s auc=%.4f" % (arm, auc))
    return "  |  ".join(msg) + "   n=%d pos=%d" % (len(yt), int(yt.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="all",
                    help="comma-separated episode names, or all")
    ap.add_argument("--assets", default="all")
    ap.add_argument("--maps", default=",".join(MAPS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    eps = list(EPISODES) if a.episodes == "all" else a.episodes.split(",")
    ass = list(ASSETS) if a.assets == "all" else a.assets.split(",")
    mps = a.maps.split(",")

    print("device %s   window %d   SIG %d" % (fa.device, WINDOW, SIG))
    print("episodes %s" % ", ".join(eps))
    t0 = time.time()
    for ep in eps:
        if ep not in EPISODES:
            print("unknown episode", ep)
            continue
        cfg = EPISODES[ep]
        print()
        print("=" * 74)
        print("%s   out=%s   onset=%s" % (ep, cfg["out"], cfg["onset"]))
        print("=" * 74, flush=True)
        for asset in ass:
            for m in mps:
                r = run_cell(ep, cfg, asset, m, force=a.force)
                print("  %-7s %-12s %s   [%.0f s]"
                      % (asset, m, r, time.time() - t0), flush=True)
    print()
    print("all done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()

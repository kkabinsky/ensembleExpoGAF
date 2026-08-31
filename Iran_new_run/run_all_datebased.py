# -*- coding: utf-8 -*-
"""
run_all_datebased.py

Date-based runner: the image is built from the raw-price window.

Run

    python run_all_datebased.py
"""
import os, sys
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fanogan_four_gaf_compare as fa
import oil_baselines_gaf as obg
import improved_tadgans_anomaly2 as tadgan
if not torch.cuda.is_available():
    fa.device = torch.device("cpu")
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

WINDOW = fa.WINDOW_SIZE
MAPS = ["exponential", "cosine", "arccosh", "arctan"]
SIG = int(os.environ.get("TADGAN_SIGNAL_SHAPE", "100"))
TADGAN_EPOCHS = int(os.environ.get("TADGAN_EPOCHS", "50"))
FORCE = os.environ.get("FORCE", "0") != "0"
BASELINES_ONLY = os.environ.get("BASELINES_ONLY", "0") != "0"
OUT = os.path.join(HERE, os.environ.get("OUT_DIR", "results"))
DATA_SUBDIR = os.environ.get("DATA_SUBDIR", "datasets")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ASSETS = {"USOIL": "USOIL_daily_final2.xlsx", "GOLD": "GOLD_daily_final2.xlsx",
          "EURUSD": "EURUSD_daily_final2.xlsx"}
# DATE-BASED scenario (overridable via env for other events, e.g. 2026)
SC = {"train_end": os.environ.get("SC_TRAIN_END", "2025-03-31"),
      "test_start": os.environ.get("SC_TEST_START", "2025-04-01"),
      "test_end": os.environ.get("SC_TEST_END", "2025-08-31"),
      "crash_onset": os.environ.get("SC_CRASH_ONSET", "2025-06-12"),
      "crash_end": os.environ.get("SC_CRASH_END", "2025-06-25")}

# ---- date-based split, monkey-patched into the baselines module ----
_MASKS = {}   # set per asset before running baselines


def date_split(prices, gafs, labels, starts):
    tr, te = _MASKS["train"], _MASKS["test"]
    trn = tr & (labels == 0)
    return gafs[trn], gafs[te], labels[te], starts[te]


obg.split_train_test = date_split


def load(asset):
    df = pd.read_excel(os.path.join(HERE, DATA_SUBDIR, ASSETS[asset]))
    c = {x.lower(): x for x in df.columns}
    d = pd.to_datetime(df[c["date"]]).reset_index(drop=True)
    p = pd.to_numeric(df[c["cp"]], errors="coerce").to_numpy(dtype=np.float64)
    return p, d


def masks(dates, prices):
    n = len(prices)
    starts = np.arange(n - WINDOW + 1)
    sd = dates.iloc[starts].reset_index(drop=True)
    tr = (sd <= pd.Timestamp(SC["train_end"])).to_numpy()
    te = ((sd >= pd.Timestamp(SC["test_start"])) & (sd <= pd.Timestamp(SC["test_end"]))).to_numpy()
    cs = int(np.where(dates >= pd.Timestamp(SC["crash_onset"]))[0].min())
    ce_arr = np.where(dates <= pd.Timestamp(SC["crash_end"]))[0]
    ce = int(ce_arr.max()) + 1
    onset = cs
    train_cut = int(np.where(dates <= pd.Timestamp(SC["train_end"]))[0].max()) + 1
    return starts, tr, te, cs, ce, onset, train_cut


def save_cell(asset, method, mapping, starts_test, y_test, scores, ypred, we=None):
    d = os.path.join(OUT, asset, method, mapping) if mapping else os.path.join(OUT, asset, method)
    os.makedirs(d, exist_ok=True)
    if we is None:
        we = np.asarray(starts_test) + WINDOW - 1
    pd.DataFrame({"window_start": starts_test, "window_end": we,
                  "label_0normal_1crash": y_test, "anomaly_score": scores,
                  "predicted_label": ypred}).to_csv(os.path.join(d, "test_scores.csv"), index=False)
    try:
        auc = float(roc_auc_score(y_test, scores))
    except Exception:
        auc = float("nan")
    pd.DataFrame([{"dataset": asset, "method": method, "mapping": mapping or "-",
                   "auc": auc, "f1": float(f1_score(y_test, ypred, zero_division=0)),
                   "n_test": int(len(y_test)), "n_crash": int(np.sum(y_test))}]).to_csv(
        os.path.join(d, "metrics_summary.csv"), index=False)
    return auc


def done(asset, method, mapping):
    d = os.path.join(OUT, asset, method, mapping) if mapping else os.path.join(OUT, asset, method)
    return (not FORCE) and os.path.isfile(os.path.join(d, "metrics_summary.csv"))


def fanogan_scores(x_train, x_test):
    loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)),
                        batch_size=fa.BATCH_SIZE, shuffle=True, drop_last=True)
    G, D, E = fa.Generator().to(fa.device), fa.Discriminator().to(fa.device), fa.Encoder().to(fa.device)
    fa.train_wgangp(G, D, loader)
    fa.train_encoder(E, G, D, loader)
    return fa.anomaly_scores(E, G, D, x_train), fa.anomaly_scores(E, G, D, x_test)


def tadgan_opquantile(asset, prices, train_cut):
    """train TadGAN on pre-event -> detection rate on train -> operating quantile."""
    stage = os.path.join(OUT, asset, "tadgan_stage")
    os.makedirs(os.path.join(stage, "models"), exist_ok=True)
    os.makedirs(os.path.join(stage, "out"), exist_ok=True)
    mu, sdv = prices[:train_cut].mean(), prices[:train_cut].std() + 1e-10
    z = (prices - mu) / sdv
    tr_csv, full_csv = os.path.join(stage, "train.csv"), os.path.join(stage, "full.csv")
    pd.DataFrame({"signal": z[:train_cut]}).to_csv(tr_csv, index=False)
    pd.DataFrame({"signal": z}).to_csv(full_csv, index=False)
    cfg = {"signal_shape": SIG, "latent_dim": 100, "hidden_dim": 40, "batch_size": 64,
           "lr": 1e-5, "n_epochs": TADGAN_EPOCHS, "n_critic": 5, "lambda_gp": 10,
           "checkpoint_interval": 10, "stride": 1, "normalize": False,
           "threshold_method": "statistical", "threshold_value": 2.0,
           "prune_false_positives": False, "generate_plots": False, "show_plots": False,
           "export_excel": False, "model_dir": os.path.join(stage, "models"),
           "output_dir": os.path.join(stage, "out"), "device": DEVICE}
    full_out = os.path.join(stage, "out", "full_scores.csv")
    if os.path.isfile(full_out) and len(pd.read_csv(full_out)) == max(0, len(z) - SIG + 1):
        res = pd.read_csv(full_out)
    else:
        tadgan.process_file(tr_csv, os.path.join(stage, "out", "tr.csv"), cfg, force_train=True)
        res = tadgan.process_file(full_csv, full_out, cfg, force_train=False)
    sc = res["anomaly_score"].to_numpy(float)
    pe = np.arange(len(sc)) + SIG - 1
    trm = pe < train_cut
    thr = np.quantile(sc[trm], 0.95)
    rate = float((sc[trm] > thr).mean()) if trm.any() else 0.05
    return float(min(max(1.0 - rate, 0.50), 0.999)), sc, pe


def run_asset(asset):
    print(f"\n{'#'*70}\n[{asset}]  test {SC['test_start']}..{SC['test_end']}  crash {SC['crash_onset']}\n{'#'*70}", flush=True)
    prices, dates = load(asset)
    starts, tr, te, cs, ce, onset, train_cut = masks(dates, prices)
    _MASKS["train"], _MASKS["test"] = tr, te
    fa.CRASH_START, fa.CRASH_END = cs, ce
    print(f"  train={int(tr.sum())} test={int(te.sum())} onset_idx={onset} crash_windows={int(((starts<=ce)&(starts+WINDOW>cs)&te).sum())}", flush=True)

    # ---- baselines (exponential) : they save themselves via obg.STOCK_NAME ----
    gaf_exp = fa.GAF_METHODS["exponential"]
    obg.STOCK_NAME = os.path.join(OUT, asset)
    for bl, fn in obg.ALL_GAF_METHODS.items():
        if done(asset, bl, "exponential"):
            continue
        r = fn(prices, "exponential", gaf_exp)
        print(f"  [baseline] {bl}: auc={r.get('auc', float('nan')):.3f}", flush=True)

    if BASELINES_ONLY:
        return

    # ---- standalone f-AnoGAN + hybrid2 (4 maps) ----
    op_q = None
    for m in MAPS:
        gaf_fn = fa.GAF_METHODS[m]
        gafs, labels, all_starts = fa.build_windows(prices, gaf_fn)
        x_tr, x_te = gafs[tr & (labels == 0)], gafs[te]
        y_te, st_te = labels[te], all_starts[te]
        if len(x_tr) < max(fa.BATCH_SIZE, 8) or len(x_te) == 0:
            continue
        if not done(asset, "standalone_fanogan", m) or not done(asset, "hybrid2", m):
            trs, tes = fanogan_scores(x_tr, x_te)
            # standalone: Q0.95
            if not done(asset, "standalone_fanogan", m):
                thr = np.quantile(trs, fa.THRESHOLD_Q)
                save_cell(asset, "standalone_fanogan", m, st_te, y_te, tes, (tes > thr).astype(int))
                print(f"  [ExpoGAF] {m}: done", flush=True)
            # hybrid2: threshold from TadGAN operating quantile
            if not done(asset, "hybrid2", m):
                if op_q is None:
                    op_q, _, _ = tadgan_opquantile(asset, prices, train_cut)
                thr2 = np.quantile(trs, op_q)
                save_cell(asset, "hybrid2", m, st_te, y_te, tes, (tes > thr2).astype(int))
                print(f"  [hybrid2] {m}: done (op_q={op_q:.3f})", flush=True)

    # ---- standalone TadGAN (pre-event) ----
    if not done(asset, "standalone_tadgan", ""):
        _, sc, pe = tadgan_opquantile(asset, prices, train_cut)
        trm = pe < train_cut
        thr = np.mean(sc[trm]) + 2 * np.std(sc[trm])
        det = (sc > thr).astype(int)
        s = np.arange(len(sc))                                 # score-window index
        lab = ((s <= ce) & (s + SIG > cs)).astype(int)         # overlap crash
        ts_idx = int(np.where(dates >= pd.Timestamp(SC["test_start"]))[0].min())
        te_end_idx = int(np.where(dates <= pd.Timestamp(SC["test_end"]))[0].max())
        tem = (pe >= ts_idx) & (pe <= te_end_idx)              # window-end in test span
        save_cell(asset, "standalone_tadgan", "", s[tem], lab[tem], sc[tem], det[tem], we=pe[tem])
        print(f"  [standalone_tadgan]: done", flush=True)


def main():
    for a in ASSETS:
        run_asset(a)
    print("\n[ALL DONE] -> results/", flush=True)


if __name__ == "__main__":
    main()

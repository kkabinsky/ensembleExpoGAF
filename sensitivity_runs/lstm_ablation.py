# -*- coding: utf-8 -*-
"""
lstm_ablation.py
================

The LSTM component of the TadGAN front end. Reviewer 4, point 6, which asks for
a component-wise ablation of TadGAN, LSTM, GAF and f-AnoGAN.

The program runs two checks, and the first one costs nothing.

Check one: is the recurrent layer recurrent?
    In `improved_tadgans_anomaly2.py` the encoder reshapes its input with

        x = x.view(-1, 1, self.signal_shape)

    so the LSTM receives a sequence of length one whose single element is the
    whole window. An LSTM run over one timestep has nothing to carry from one
    step to the next: its forget and output gates fire once, on the initial
    zero state, and the result is a gated affine function of that one vector.
    The decoder does the same, on a sequence of length one in latent space.
    Both are declared bidirectional, which over a single step means the two
    directions see the same input.

    This check verifies that claim on the code as it stands rather than
    asserting it: it feeds the encoder a batch, reverses the window along its
    time axis, and compares. A layer that used the order of the observations
    could not return the same thing for both. It also confirms the sequence
    length reaching the LSTM is one.

Check two: does replacing it change the result?
    Three front ends are trained on the same windows with the same seeds and
    the same budget, and their reconstruction errors are compared as anomaly
    scores:

        lstm    the published encoder and decoder, unchanged
        gru     the same shapes with the recurrent cell swapped for a GRU
        linear  the same shapes with the recurrent cell replaced by an affine
                layer of matched output width, which is what check one predicts
                the LSTM is already equivalent to

    If the three agree within the spread across seeds, the recurrent unit is
    not contributing and the manuscript should say so.

    The budget here is smaller than the published one, and deliberately so: the
    question is whether the three front ends differ from each other under
    identical treatment, not what the best attainable score is. The setting is
    written into every row.

Stopping and resuming
    Every finished cell is written immediately. Interrupt it and rerun the same
    command: it continues from the next unfinished cell.

Run
    python lstm_ablation.py --check-only        the free structural check
    python lstm_ablation.py --seeds 3 --epochs 40
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "Iran_new_run")
OUT = os.path.join(HERE, "output")

os.environ.setdefault("ALLOW_CPU", "1")
sys.path.insert(0, RUN)
import improved_tadgans_anomaly2 as td                           # noqa: E402

SIG = 100          # the window the TadGAN front end reads, as published
WINDOW = 32        # the window the manuscript scores and labels
LATENT = 20
RESULT_DIR = {"covid": "results_covid", "russia": "results_russia",
              "chinese": "results_chinese", "iran2025": "results"}
ASSETS = {"USOIL": "USOIL_daily_final2.xlsx",
          "GOLD": "GOLD_daily_final2.xlsx",
          "EURUSD": "EURUSD_daily_final2.xlsx"}
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
}


# ------------------------------------------------------------------ variants
class Encoder(nn.Module):
    """The published encoder, with the recurrent cell chosen at construction."""

    def __init__(self, kind, signal_shape=SIG, latent_dim=LATENT):
        super().__init__()
        self.kind = kind
        self.signal_shape = signal_shape
        self.latent_dim = latent_dim
        if kind == "lstm":
            self.rec = nn.LSTM(signal_shape, latent_dim, num_layers=1,
                               bidirectional=True, batch_first=True)
        elif kind == "gru":
            self.rec = nn.GRU(signal_shape, latent_dim, num_layers=1,
                              bidirectional=True, batch_first=True)
        elif kind == "linear":
            self.rec = nn.Linear(signal_shape, latent_dim * 2)
        else:
            raise ValueError(kind)
        self.dense = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, self.signal_shape).float()
        if self.kind == "linear":
            h = torch.tanh(self.rec(x))
        else:
            h, _ = self.rec(x)
        return self.dense(h)


class Decoder(nn.Module):
    def __init__(self, kind, signal_shape=SIG, latent_dim=LATENT):
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.rec = nn.LSTM(latent_dim, latent_dim * 2, num_layers=2,
                               bidirectional=True, batch_first=True)
        elif kind == "gru":
            self.rec = nn.GRU(latent_dim, latent_dim * 2, num_layers=2,
                              bidirectional=True, batch_first=True)
        elif kind == "linear":
            self.rec = nn.Linear(latent_dim, latent_dim * 4)
        else:
            raise ValueError(kind)
        self.dense = nn.Linear(latent_dim * 4, signal_shape)

    def forward(self, x):
        if self.kind == "linear":
            h = torch.tanh(self.rec(x))
        else:
            h, _ = self.rec(x)
        return self.dense(h)


def count(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------ structural check
def structural_check():
    """Show what the recurrent layer actually receives, on the published code."""
    print("=" * 80)
    print("Check one: is the recurrent layer recurrent?")
    print("=" * 80)
    torch.manual_seed(0)
    pub = td.Encoder(signal_shape=SIG, latent_dim=LATENT).eval()
    x = torch.randn(8, SIG)

    seen = {}

    def hook(mod, inp, out):
        seen["shape"] = tuple(inp[0].shape)

    h = pub.lstm.register_forward_hook(hook)
    with torch.no_grad():
        a = pub(x)
        b = pub(torch.flip(x, dims=[1]))
    h.remove()

    print("  input to the LSTM has shape %s" % (seen["shape"],))
    print("  sequence length reaching the recurrent cell: %d"
          % seen["shape"][1])
    same = bool(torch.allclose(a, b))
    print("  reversing the window changes the output: %s"
          % ("no" if same else "yes"))
    print("    this on its own proves nothing: an affine layer also changes")
    print("    its output when its input vector is reversed. The decisive")
    print("    fact is the sequence length on the line above.")
    print()
    if seen["shape"][1] == 1:
        print("  The sequence length is one. The cell fires once, on the zero")
        print("  initial state, so nothing is carried between timesteps and")
        print("  the layer is a gated affine map of the whole window. The")
        print("  bidirectional flag has no effect over a single step.")
    else:
        print("  The sequence length is greater than one, so the layer is")
        print("  genuinely recurrent and the reading below does not apply.")
    print()
    print("  parameters, published encoder %d, decoder %d"
          % (count(pub), count(td.Decoder(signal_shape=SIG,
                                          latent_dim=LATENT))))
    for k in ("lstm", "gru", "linear"):
        print("  parameters, %-6s encoder %7d   decoder %7d"
              % (k, count(Encoder(k)), count(Decoder(k))))
    return seen["shape"][1], same


# --------------------------------------------------------------------- data
def windows(cfg, asset):
    """Everything one cell needs, on the window positions the manuscript uses.

    Each 100-step window is scaled to [-1, 1] on its own values. Scaling the
    whole series at once instead would leave the level of the price inside the
    window, and that level drifts away from the training range as the test
    period advances; a reconstruction model then scores a window by how far it
    has drifted rather than by what happened in it. Written that way on a first
    pass, this comparison returned an AUC of exactly 1.000 on one asset and
    exactly 0.000 on another, which is a clock and not a detector. The GAF
    stage downstream scales each window on its own for the same reason.

    Evaluation follows the pipeline rather than the 100-step window. The
    per-window reconstruction error becomes a per-step score, and the 32-step
    score window that the manuscript labels is reduced to one number by its
    mean and by its maximum. The window positions and the labels come from the
    saved run, so the cells line up with the published tables.
    """
    df = pd.read_excel(os.path.join(RUN, cfg["data"], ASSETS[asset]))
    c = {x.lower(): x for x in df.columns}
    price = pd.to_numeric(df[c["cp"]], errors="coerce").to_numpy(float)
    dates = pd.to_datetime(df[c["date"]]).reset_index(drop=True)

    starts = np.arange(len(price) - SIG + 1)
    raw = np.stack([price[s:s + SIG] for s in starts])
    lo = raw.min(axis=1, keepdims=True)
    hi = raw.max(axis=1, keepdims=True)
    span = np.where(hi > lo, hi - lo, 1.0)
    X = (2.0 * (raw - lo) / span - 1.0).astype(np.float32)

    # score index i corresponds to the price index i + SIG - 1
    sd = dates.iloc[starts + SIG - 1].reset_index(drop=True)
    train = (sd <= pd.Timestamp(cfg["train_end"])).to_numpy()

    sc = os.path.join(RUN, RESULT_DIR[cfg["key"]], asset, "hybrid2",
                      "exponential", "test_scores.csv")
    if not os.path.isfile(sc):
        return None
    d = pd.read_csv(sc).sort_values("window_start")
    w0 = d["window_start"].to_numpy(int)
    y = d["label_0normal_1crash"].to_numpy(int)
    si = w0 - (SIG - 1)
    keep = (si >= 0) & (si + WINDOW <= len(starts))
    if keep.sum() == 0:
        return None
    return X, train, si[keep], y[keep]


def reduce_windows(step_scores, si, how):
    """One number per labelled window, from the per-step scores."""
    out = np.empty(len(si))
    for k, s in enumerate(si):
        seg = step_scores[s:s + WINDOW]
        out[k] = seg.mean() if how == "mean" else seg.max()
    return out


def fit_step_scores(kind, x_tr, x_all, epochs, seed, lr=1e-4,
                    batch=64):
    """Fit the encoder and decoder as a reconstruction pair and score by the
    reconstruction error, which is the part of the TadGAN score the ablation
    is about. The critics are left out on purpose: they are common to all
    three variants and would add noise without bearing on the question."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc, dec = Encoder(kind), Decoder(kind)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=lr, betas=(0.5, 0.999))
    xt = torch.tensor(x_tr, dtype=torch.float32)
    n = len(xt)
    enc.train()
    dec.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            xb = xt[perm[i:i + batch]]
            opt.zero_grad()
            out = dec(enc(xb)).view(-1, SIG)
            nn.functional.mse_loss(out, xb).backward()
            opt.step()
    enc.eval()
    dec.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_all), 512):
            xb = torch.tensor(x_all[i:i + 512], dtype=torch.float32)
            rec = dec(enc(xb)).view(-1, SIG)
            out.append(((xb - rec) ** 2).mean(dim=1).numpy())
    return np.concatenate(out).astype(float)


def auc(y, s):
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--kinds", default="lstm,gru,linear")
    ap.add_argument("--episodes", default="covid,russia,chinese,iran2025")
    ap.add_argument("--assets", default="USOIL,GOLD,EURUSD")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)

    seq_len, order_matters = structural_check()
    pd.DataFrame([{"sequence_length_into_recurrent_cell": seq_len,
                   "reversing_the_window_changes_the_output":
                       not order_matters}]).to_csv(
        os.path.join(OUT, "lstm_structural_check.csv"), index=False)
    if a.check_only:
        return

    kinds = [x.strip() for x in a.kinds.split(",") if x.strip()]
    eps = [x.strip() for x in a.episodes.split(",") if x.strip()]
    assets = [x.strip() for x in a.assets.split(",") if x.strip()]

    csv = os.path.join(OUT, "lstm_ablation%s.csv" % a.tag)
    rows, done = [], set()
    if os.path.exists(csv):
        try:
            prev = pd.read_csv(csv)
            rows = prev.to_dict("records")
            done = {(str(r["kind"]), str(r["episode"]), str(r["asset"]),
                     int(r["seed"])) for _, r in prev.iterrows()}
            print("found %d finished cells; continuing from there" % len(done),
                  flush=True)
        except Exception as e:
            print("could not read earlier results (%s); starting over" % e)

    def flush():
        tmp = csv + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, csv)

    print("=" * 80)
    print("Check two: does replacing the recurrent cell change the result?")
    print("=" * 80)
    print("window %d, epochs %d, seeds %d, reconstruction error as the score"
          % (SIG, a.epochs, a.seeds), flush=True)
    print()

    total = len(kinds) * len(eps) * len(assets) * a.seeds
    t0 = time.time()
    n = 0
    for ep in eps:
        cfg = dict(EPISODES[ep])
        cfg["key"] = ep
        for asset in assets:
            built = windows(cfg, asset)
            if built is None:
                print("%-9s %-7s skipped, no saved run to line up with"
                      % (ep, asset), flush=True)
                continue
            X, train, si, y = built
            if train.sum() < 64 or y.sum() in (0, len(y)):
                print("%-9s %-7s skipped, not enough data" % (ep, asset),
                      flush=True)
                continue
            # a detector that scores a window by its position and nothing else.
            # An arm that does not beat this is reporting the calendar.
            if ("time_index", ep, asset, 0) not in done:
                rows.append({"kind": "time_index", "reduce": "position",
                             "episode": ep, "asset": asset, "seed": 0,
                             "auc": auc(y, np.arange(len(y), dtype=float)),
                             "n_train": int(train.sum()), "n_test": len(y),
                             "prevalence": float(y.mean()), "epochs": 0,
                             "window": WINDOW, "signal": SIG})
                flush()
            for kind in kinds:
                for sd in range(a.seeds):
                    n += 1
                    if (kind, ep, asset, sd) in done:
                        continue
                    step = fit_step_scores(kind, X[train], X, a.epochs, sd)
                    line = []
                    for how in ("mean", "max"):
                        v = auc(y, reduce_windows(step, si, how))
                        rows.append({"kind": kind, "reduce": how,
                                     "episode": ep, "asset": asset,
                                     "seed": sd, "auc": v,
                                     "n_train": int(train.sum()),
                                     "n_test": len(y),
                                     "prevalence": float(y.mean()),
                                     "epochs": a.epochs, "window": WINDOW,
                                     "signal": SIG})
                        line.append("%s %.4f" % (how, v))
                    flush()
                    print("%-7s %-9s %-7s seed %d   %s   [%d/%d  %.0f min]"
                          % (kind, ep, asset, sd, "  ".join(line), n, total,
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
    print("Mean AUC by front end, over the cells and seeds")
    print("=" * 80)
    tab = r.pivot_table(index="kind", columns="reduce", values="auc",
                        aggfunc="mean")
    print(tab.round(4).to_string())
    r = r[r["reduce"] != "position"]
    g = r.groupby("kind").agg(cells=("auc", "size"),
                              mean=("auc", "mean"), sd=("auc", "std"))

    seed_sd = float(r.groupby(["kind", "episode", "asset"])["auc"].std().mean())
    spread = float(g["mean"].max() - g["mean"].min())
    print()
    print("  spread between the three front ends          %.4f" % spread)
    print("  mean spread between seeds of the same one    %.4f" % seed_sd)
    if spread < seed_sd:
        print()
        print("  The three front ends differ by less than a change of seed")
        print("  does. On this evidence the recurrent unit is not what makes")
        print("  the front end work.")

    print()
    print("=" * 80)
    print("Paired against the published lstm front end")
    print("=" * 80)
    piv = r[r["reduce"] == "mean"].pivot_table(
        index=["episode", "asset", "seed"], columns="kind", values="auc")
    if "lstm" in piv.columns:
        from scipy.stats import binomtest
        print("  %-10s %6s %11s %10s %9s"
              % ("against", "cells", "mean diff", "wins", "sign p"))
        for k in [c for c in piv.columns if c != "lstm"]:
            d = (piv[k] - piv["lstm"]).dropna()
            if d.empty:
                continue
            w = int((d > 0).sum())
            print("  %-10s %6d %+11.4f %5d/%-3d %9.3f"
                  % (k, len(d), d.mean(), w, len(d),
                     binomtest(w, len(d), 0.5).pvalue))
    print()
    print("Output file", csv)


if __name__ == "__main__":
    main()

"""
dm_test_mappings.py
===================

Diebold-Mariano between the four angular mappings, under both scorers.

Windows are 32 observations long and advance by one, so adjacent windows share
31 of their 32 values and the loss differential is strongly autocorrelated. A
plain variance understates the standard error, so every statistic is computed
twice: once at lag 0 as the submitted version did, and once with a Newey-West
estimator at lag 31. Holm correction is applied over the six pairs.

Two scorers, because they disagree:
    f-AnoGAN     one-class, from the stored scores, nothing retrained
    CNN          supervised, leave-one-episode-out

The script also reports, per episode, how many pairs would look significant at
lag 0 but do not survive at lag 31. That difference is the size of the error
made by ignoring the overlap.

Run
    python dm_test_mappings.py
"""
import argparse
import io
import itertools
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "Iran_new_run")
OUT = os.path.join(HERE, "output")
CACHE = os.path.join(OUT, "gaf_dataset.npz")

EPISODES = {"COVID-19": "results_covid", "Russia-Ukraine": "results_russia",
            "Chinese": "results_chinese", "Iran 2025": "results"}
ASSETS = ["USOIL", "GOLD", "EURUSD"]
ORDER = ["cosine", "arctan", "arccosh", "exponential"]
WINDOW = 32
OVERLAP_LAG = WINDOW - 1          # the 31 observations adjacent windows share


# ---------------------------------------------------------------------------
def rank01(s):
    r = pd.Series(s).rank(method="average").to_numpy(float)
    return (r - 0.5) / len(r)


def nw_var(d, lag):
    """Newey-West variance of the mean, with Bartlett weights."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 3:
        return np.nan
    x = d - d.mean()
    g0 = float(np.dot(x, x) / T)
    s = g0
    for k in range(1, min(lag, T - 1) + 1):
        gk = float(np.dot(x[k:], x[:-k]) / T)
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    s = max(s, 1e-12)
    return s / T


def dm_stat(loss_a, loss_b, lag):
    """DM > 0 means A loses more than B, so B is better."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return np.nan, np.nan
    v = nw_var(d, lag)
    if not np.isfinite(v) or v <= 0:
        return np.nan, np.nan
    stat = float(d.mean() / np.sqrt(v))
    # two-sided standard normal
    from math import erfc, sqrt
    p = float(erfc(abs(stat) / sqrt(2.0)))
    return stat, p


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min((m - rank) * p[i], 1.0))
        adj[i] = prev
    return adj


# ---------------------------------------------------------------------------
def losses_fanogan():
    """Per-window f-AnoGAN loss from the stored scores; nothing retrained.
    Returned per episode so the test can also be run one episode at a time."""
    per = {ep: {m: [] for m in ORDER} for ep in EPISODES}
    for ep, sub in EPISODES.items():
        for asset in ASSETS:
            cells = {}
            ok = True
            for m in ORDER:
                p = os.path.join(RUN, sub, asset, "hybrid2", m, "test_scores.csv")
                if not os.path.exists(p):
                    ok = False
                    break
                cells[m] = pd.read_csv(p).sort_values("window_start")
            if not ok:
                continue
            y = cells[ORDER[0]]["label_0normal_1crash"].to_numpy(int)
            if y.sum() in (0, len(y)):
                continue
            for m in ORDER:
                s = pd.to_numeric(cells[m]["anomaly_score"],
                                  errors="coerce").to_numpy(float)
                per[ep][m].append((rank01(s) - y) ** 2)
    return {ep: {m: np.concatenate(v) for m, v in d.items() if v}
            for ep, d in per.items() if any(d.values())}


def pool(per):
    """Pool every episode together."""
    out = {}
    for m in ORDER:
        parts = [d[m] for d in per.values() if m in d]
        if parts:
            out[m] = np.concatenate(parts)
    return out


# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, k=7, c1=32, c2=64):
        super().__init__()
        p = k // 2
        self.f = nn.Sequential(
            nn.Conv2d(1, c1, k, padding=p), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, k, padding=p), nn.ELU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(c2, 1), nn.Flatten(0))

    def forward(self, x):
        return self.f(x)


def losses_cnn(seeds, epochs):
    """Per-window CNN loss under leave-one-episode-out.
    Probabilities are averaged over seeds first, then the loss is taken."""
    if not os.path.exists(CACHE):
        print("not found:", CACHE, "- run cnn_architecture_sweep.py first")
        return None
    z = np.load(CACHE, allow_pickle=True)
    data = {m: dict(X=z[f"{m}_X"], y=z[f"{m}_y"], ep=z[f"{m}_ep"]) for m in ORDER}
    out = {m: [] for m in ORDER}
    per = {}
    for held in EPISODES:
        for m in ORDER:
            X, y, ep = data[m]["X"], data[m]["y"], data[m]["ep"]
            te = ep == held
            tr = ~te
            if te.sum() == 0 or y[te].sum() in (0, int(te.sum())):
                continue
            probs = []
            for sd in range(seeds):
                torch.manual_seed(sd)
                np.random.seed(sd)
                mdl = Net()
                opt = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
                pos = float(y[tr].sum())
                w = torch.tensor(max(len(y[tr]) - pos, 1.0) / max(pos, 1.0),
                                 dtype=torch.float32)
                lf = nn.BCEWithLogitsLoss(pos_weight=w)
                xt = torch.tensor(X[tr], dtype=torch.float32)
                yt = torch.tensor(y[tr], dtype=torch.float32)
                mdl.train()
                for _ in range(epochs):
                    perm = torch.randperm(len(xt))
                    for i in range(0, len(xt), 64):
                        idx = perm[i:i + 64]
                        opt.zero_grad()
                        lf(mdl(xt[idx]), yt[idx]).backward()
                        opt.step()
                mdl.eval()
                with torch.no_grad():
                    probs.append(torch.sigmoid(
                        mdl(torch.tensor(X[te], dtype=torch.float32))).numpy())
            pm = np.mean(probs, axis=0)
            out[m].append((rank01(pm) - y[te]) ** 2)
            per.setdefault(held, {})[m] = (rank01(pm) - y[te]) ** 2
            print("    %-12s held-out %-16s done" % (m, held), flush=True)
    return per


# ---------------------------------------------------------------------------
def report(losses, tag):
    print()
    print("=" * 92)
    print("DM test : %s" % tag)
    print("=" * 92)
    n = len(next(iter(losses.values())))
    print("Windows used %d; mean loss per mapping" % n)
    for m in ORDER:
        if m in losses:
            print("    %-12s %.4f" % (m, losses[m].mean()))
    rows = []
    for a, b in itertools.combinations([m for m in ORDER if m in losses], 2):
        s0, p0 = dm_stat(losses[a], losses[b], 0)
        s1, p1 = dm_stat(losses[a], losses[b], OVERLAP_LAG)
        rows.append({"A": a, "B": b, "loss_A": losses[a].mean(),
                     "loss_B": losses[b].mean(),
                     "DM_lag0": s0, "p_lag0": p0,
                     "DM_lag31": s1, "p_lag31": p1,
                     "better": b if s1 > 0 else a})
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    r["p_holm_lag31"] = holm(r["p_lag31"].fillna(1.0).to_numpy()).round(4)
    print()
    print(r.round(4).to_string(index=False))
    print()
    sig = r[r["p_holm_lag31"] < 0.05]
    print("Pairs significant after Holm correction at lag 31: %d of %d"
          % (len(sig), len(r)))
    if len(sig):
        for _, x in sig.iterrows():
            print("    %s against %s : %s better,  p = %.4g"
                  % (x["A"], x["B"], x["better"], x["p_holm_lag31"]))
    print()
    bad = r[(r["p_lag0"] < 0.05) & (r["p_lag31"] >= 0.05)]
    print("Pairs significant at lag 0 but not at lag 31: %d" % len(bad))
    if len(bad):
        print("    that is the size of the error from ignoring the overlap")
    return r



def report_per_episode(per, tag):
    """DM per episode, with Holm correction over the six pairs inside it."""
    print()
    print("=" * 92)
    print("DM by episode: %s" % tag)
    print("=" * 92)
    allrows = []
    for ep in sorted(per):
        L = per[ep]
        keys = [m for m in ORDER if m in L]
        if len(keys) < 2:
            continue
        n = len(L[keys[0]])
        neff = n / WINDOW
        rows = []
        for a, b in itertools.combinations(keys, 2):
            s1, p1 = dm_stat(L[a], L[b], OVERLAP_LAG)
            rows.append({"episode": ep, "A": a, "B": b,
                         "loss_A": L[a].mean(), "loss_B": L[b].mean(),
                         "DM_lag31": s1, "p": p1,
                         "better": (b if (s1 or 0) > 0 else a)})
        r = pd.DataFrame(rows)
        r["p_holm"] = holm(r["p"].fillna(1.0).to_numpy()).round(4)
        print()
        print("--- %s   %d windows, about %.0f independent ---" % (ep, n, neff))
        print(r[["A", "B", "loss_A", "loss_B", "DM_lag31", "p",
                 "p_holm", "better"]].round(4).to_string(index=False))
        sig = r[r["p_holm"] < 0.05]
        print("    significant after Holm within this episode: %d of %d"
              % (len(sig), len(r)))
        allrows.append(r)
    if not allrows:
        return pd.DataFrame()
    full = pd.concat(allrows, ignore_index=True)
    print()
    print("=" * 92)
    print("Summary of the exponential pairs")
    print("=" * 92)
    e = full[(full["A"] == "exponential") | (full["B"] == "exponential")]
    piv = e.pivot_table(index="episode",
                        columns=e.apply(lambda x: x["A"] if x["B"] == "exponential"
                                        else x["B"], axis=1),
                        values="DM_lag31")
    print("DM statistic; positive favours exponential")
    print(piv.round(3).to_string())
    print()
    print("Pairs significant in total: %d of %d tests"
          % (int((full["p_holm"] < 0.05).sum()), len(full)))
    print("Note: %d tests across episodes; Holm over all of them would be stricter"
          % len(full))
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="both",
                    choices=["both", "fanogan", "cnn"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.only in ("both", "fanogan"):
        per = losses_fanogan()
        lf = pool(per)
        r = report(lf, "f-AnoGAN (stored scores, nothing retrained)")
        pe = report_per_episode(per, "f-AnoGAN")
        if not pe.empty:
            pe.to_csv(os.path.join(OUT, "dm_fanogan_per_episode.csv"), index=False)
        if r is not None and not r.empty:
            r.to_csv(os.path.join(OUT, "dm_fanogan_mappings.csv"), index=False)

    if a.only in ("both", "cnn"):
        print()
        print("training the CNN ...", flush=True)
        perc = losses_cnn(a.seeds, a.epochs)
        if perc:
            lc = pool(perc)
            r = report(lc, "supervised CNN, k7 c32-64 elu, "
                           "leave-one-episode-out")
            pe = report_per_episode(perc, "CNN k7 c32-64 elu")
            if not pe.empty:
                pe.to_csv(os.path.join(OUT, "dm_cnn_per_episode.csv"), index=False)
            if r is not None and not r.empty:
                r.to_csv(os.path.join(OUT, "dm_cnn_mappings.csv"), index=False)
    print()
    print("Output written to", OUT)


if __name__ == "__main__":
    main()

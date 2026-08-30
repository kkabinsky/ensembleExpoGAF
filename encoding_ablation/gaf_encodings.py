# -*- coding: utf-8 -*-
"""
gaf_encodings.py
================

Named gaf_encodings rather than encodings because Python imports a standard
library module of that name at start-up and a local file would shadow it.

The four angular mappings, the image builder, the data split and the metrics.

Nothing in this file trains anything, so every number it produces is exact:
run it twice on any machine and the output is identical to the last digit.
The parts of the experiment that do train live in `models.py`.

The mappings take a window that has been scaled to [0, 1] and return an angle
per observation; the image is cos(phi_i + phi_j), the Gramian angular field.
They differ in how much of the [0, pi] interval they reach and in where along
the window range they place their angular resolution, and those two properties
turn out to matter more than anything else in this experiment.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "Iran_new_run")

IMG = 32
W_REF = 32

EPISODES = {"COVID-19": ("results_covid", "datasets"),
            "Russia-Ukraine": ("results_russia", "datasets"),
            "Chinese": ("results_chinese", "datasets"),
            "Iran 2025": ("results", "datasets")}
ASSETS = {"USOIL": "USOIL_daily_final2.xlsx",
          "GOLD": "GOLD_daily_final2.xlsx",
          "EURUSD": "EURUSD_daily_final2.xlsx"}

ORDER = ["cosine", "arctan", "arccosh", "exponential"]
MAPS = {
    "cosine":      lambda s: np.arccos(np.clip(2.0 * s - 1.0, -1.0, 1.0)),
    "arctan":      lambda s: np.arctan(s),
    "arccosh":     lambda s: np.arccosh(1.0 + s),
    "exponential": lambda s: np.pi * (np.exp(s) - 1.0) / (np.e - 1.0),
}


def minmax01(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def resize_1d(x, n):
    if len(x) == n:
        return x
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def gaf_image(series, f):
    phi = f(minmax01(resize_1d(np.asarray(series, float), IMG)))
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def mapping_profile():
    """What each mapping does to a window, before any model sees it.

    Returned columns: the angular interval reached, the share of the [-1, 1]
    image range occupied, the standard deviation of the image values, and the
    angular derivative at the bottom and the top of the window range. The last
    pair is the property the exponential map is named for: it is the only one
    whose resolution rises with the value being encoded.
    """
    s = np.linspace(0.0, 1.0, IMG)
    rows = []
    for m in ORDER:
        phi = MAPS[m](s)
        g = np.cos(phi[:, None] + phi[None, :])
        d = np.gradient(phi, s)
        rows.append({"mapping": m, "phi_min": phi.min(), "phi_max": phi.max(),
                     "image_min": g.min(), "image_max": g.max(),
                     "range_used": (g.max() - g.min()) / 2.0,
                     "contrast": float(g.std()),
                     "dphi_low": float(d[1]), "dphi_high": float(d[-2])})
    return pd.DataFrame(rows)


def crash_index_range(starts, labels):
    pos = starts[labels == 1]
    if len(pos) == 0:
        return None
    return int(pos.min()) + W_REF - 1, int(pos.max())


def build(window=W_REF):
    """Images, labels and episode tags for the four mappings.

    The window positions and the labels are read from the saved run so that
    this experiment sits on exactly the windows the manuscript reports, and the
    labels are rebuilt by the same overlap rule.
    """
    data = {m: {"X": [], "y": [], "ep": [], "asset": []} for m in ORDER}
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
            rng = crash_index_range(st, d["label_0normal_1crash"].to_numpy(int))
            if rng is None:
                continue
            cs, ce = rng
            keep = (st + window <= len(price)) & (st >= 0)
            st_w = st[keep]
            if len(st_w) == 0:
                continue
            y_w = ((st_w <= ce) & (st_w + window > cs)).astype(int)
            for m in ORDER:
                f = MAPS[m]
                for s0, lab in zip(st_w, y_w):
                    data[m]["X"].append(gaf_image(price[s0:s0 + window], f))
                    data[m]["y"].append(int(lab))
                    data[m]["ep"].append(ep)
                    data[m]["asset"].append(asset)
    for m in ORDER:
        if not data[m]["X"]:
            return None
        data[m]["X"] = np.stack(data[m]["X"])[:, None, :, :]
        data[m]["y"] = np.asarray(data[m]["y"], int)
        data[m]["ep"] = np.asarray(data[m]["ep"], object)
        data[m]["asset"] = np.asarray(data[m]["asset"], object)
    return data


def auc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def average_precision(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    yy = y[o]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def linear_probe(x_tr, y_tr, x_te, ridge=1.0):
    """Ridge least squares on the flattened image.

    This is the smallest discriminative reader there is. It has no iterations
    and no seed, so it separates the question "is the information present for a
    reader that is told the answer" from any property of a particular network.
    """
    a = x_tr.reshape(len(x_tr), -1)
    b = x_te.reshape(len(x_te), -1)
    mu, sd = a.mean(0), a.std(0) + 1e-8
    a, b = (a - mu) / sd, (b - mu) / sd
    a = np.hstack([a, np.ones((len(a), 1))])
    b = np.hstack([b, np.ones((len(b), 1))])
    w = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]),
                        a.T @ (y_tr.astype(float) * 2.0 - 1.0))
    return b @ w


def mean_distance(x_tr_normal, x_te):
    """Distance to the mean of the normal training images.

    This is the smallest one-class reader there is, and it is what a
    reconstruction model collapses to when its decoder can only return the
    average of what it was shown. It is the baseline any f-AnoGAN has to beat
    to justify itself.
    """
    m = x_tr_normal.reshape(len(x_tr_normal), -1).mean(0)
    b = x_te.reshape(len(x_te), -1)
    return np.sqrt(((b - m) ** 2).sum(1))

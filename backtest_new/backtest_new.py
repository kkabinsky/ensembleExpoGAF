# -*- coding: utf-8 -*-
"""
backtest_new.py

Financial evaluation from the saved per-window alarms.

Run

    python backtest_new.py
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "Iran_new_run")
OUT = os.path.join(HERE, "output")

RET_CLIP = 0.60
MAPPING = "exponential"
MAJORITY = 5
CONTROL = "Normal 2019"

# left to right in Figure 3
COMPONENTS = [("IF", "isolation_forest"), ("OCS", "one_class_svm"),
              ("DSV", "deep_svdd"), ("DAG", "dagmm"),
              ("OMNI", "omnianomaly"), ("USAD", "usad"), ("TR", "tranad"),
              ("AT", "anomaly_transformer"), ("EXP", "standalone_fanogan")]
METHODS = [m for m, _ in COMPONENTS] + ["ENS", "ENST"]
FULL = {"IF": "Isolation Forest", "OCS": "One-Class SVM", "DSV": "Deep SVDD",
        "DAG": "DAGMM", "OMNI": "OmniAnomaly", "USAD": "USAD", "TR": "TranAD",
        "AT": "Anomaly Transformer", "EXP": "ExpoGAF-AnoNet",
        "ENS": "ENS-H majority 5 of 9", "ENST": "EnsembleExpoGAF"}
# ENST is EnsembleExpoGAF, the ensemble the framework proposes, read from the
# aligned file.
#
# ENS is the unweighted 5-of-9 majority alarm. It is rebuilt from the saved
# alarms because the reference check needs a majority column. It is left out of the
# tables and the figures unless --keep-majority is given, since the majority
# alarm is a diagnostic rather than a method the paper proposes.
ALIGNED = os.path.join("ensembleExpoGAF", "data",
                       "aligned_hard_predictions_10methods.csv")
EVENTS = ["COVID-19", "Russia--Ukraine", "Chinese", "Iran 2025", "Iran 2026",
          CONTROL]
EPISODE_DIR = {"COVID-19": "results_covid", "Russia--Ukraine": "results_russia",
               "Chinese": "results_chinese", "Iran 2025": "results",
               "Iran 2026": "results_2026",
               CONTROL: os.path.join("covid_normal", "results")}
PRICE_DIR = {"Iran 2025": "datasets", "Iran 2026": "datasets_2026",
             "COVID-19": "datasets_2026", "Russia--Ukraine": "datasets_2026",
             "Chinese": "datasets_2026", CONTROL: "datasets"}
ASSETS = ["USOIL", "GOLD", "EURUSD"]
FILES = {"USOIL": "USOIL_daily_final2.xlsx", "GOLD": "GOLD_daily_final2.xlsx",
         "EURUSD": "EURUSD_daily_final2.xlsx"}
ONSET = {"COVID-19": "2020-02-20", "Russia--Ukraine": "2022-02-11",
         "Chinese": "2023-08-07", "Iran 2025": "2025-06-12",
         "Iran 2026": "2026-06-12"}

PUBLISHED = {
    ("COVID-19", "USOIL"): (-78.9, -1.5, -1.7),
    ("COVID-19", "GOLD"): (26.3, -10.2, -10.9),
    ("COVID-19", "EURUSD"): (5.4, -3.7, -1.8),
    ("Russia--Ukraine", "USOIL"): (16.4, -5.6, -1.5),
    ("Russia--Ukraine", "GOLD"): (-6.3, -0.3, -0.2),
    ("Russia--Ukraine", "EURUSD"): (-12.3, 6.5, 0.5),
    ("Chinese", "USOIL"): (1.2, -9.9, 0.8),
    ("Chinese", "GOLD"): (0.7, 1.8, 0.0),
    ("Chinese", "EURUSD"): (-3.8, -2.9, -0.3),
    ("Iran 2025", "USOIL"): (-4.7, 2.8, 1.5),
    ("Iran 2025", "GOLD"): (28.5, -0.1, 0.2),
    ("Iran 2025", "EURUSD"): (3.7, -1.4, 0.0),
    ("Iran 2026", "USOIL"): (-31.3, 5.0, 0.0),
    ("Iran 2026", "GOLD"): (-9.5, 2.9, 2.9),
    ("Iran 2026", "EURUSD"): (-2.4, -0.2, 0.0),
    (CONTROL, "USOIL"): (-5.8, 9.9, 4.0),
    (CONTROL, "GOLD"): (4.4, -1.4, 2.5),
    (CONTROL, "EURUSD"): (-2.7, 1.3, -0.7),
}

_px = {}


def prices(event, asset):
    key = (event, asset)
    if key not in _px:
        df = pd.read_excel(os.path.join(RUN, PRICE_DIR[event], FILES[asset]))
        c = {x.lower(): x for x in df.columns}
        _px[key] = (pd.to_numeric(df[c["cp"]], errors="coerce").to_numpy(float),
                    pd.to_datetime(df[c["date"]]).reset_index(drop=True))
    return _px[key]


def read_cell(event, asset):
    """The nine saved alarm series for one episode and asset, joined on the
    window endpoint. Returns the endpoints, the alarms and the scores."""
    base = os.path.join(RUN, EPISODE_DIR[event], asset)
    frame = None
    for short, folder in COMPONENTS:
        p = os.path.join(base, folder, MAPPING, "test_scores.csv")
        if not os.path.isfile(p):
            return None
        s = pd.read_csv(p).sort_values("window_start")
        end = (s["window_end"].to_numpy(int) if "window_end" in s.columns
               else s["window_start"].to_numpy(int) + 31)
        one = pd.DataFrame({"window_end": end,
                            "alarm_%s" % short: s["predicted_label"].to_numpy(int),
                            "score_%s" % short: s["anomaly_score"].to_numpy(float)})
        frame = one if frame is None else frame.merge(one, on="window_end",
                                                      how="inner")
    frame = frame.sort_values("window_end").reset_index(drop=True)
    a = read_trained_ensemble(event, asset)
    if a is not None:
        frame = frame.merge(a, on="window_end", how="left")
        frame["alarm_ENST"] = frame["alarm_ENST"].fillna(0).astype(int)
        frame["score_ENST"] = frame["score_ENST"].fillna(0.0)
    return frame


_aligned = None


def read_trained_ensemble(event, asset):
    """The trained ensemble alarm, keyed to the same window endpoints."""
    global _aligned
    if _aligned is None:
        p = os.path.join(RUN, ALIGNED)
        _aligned = pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()
    if _aligned.empty:
        return None
    g = _aligned[(_aligned["event"] == event) & (_aligned["asset"] == asset)]
    if g.empty:
        return None
    return pd.DataFrame({"window_end": g["window_start"].to_numpy(int) + 31,
                         "alarm_ENST": g["pred_ENS"].to_numpy(int),
                         "score_ENST": g["prob_ENS"].to_numpy(float)})


def null_percentile(ret, pos, delta, draws, rng):
    """Where the detector's excess return sits among random exit dates that
    spend exactly as long in the market.

    Without this the comparison is confounded. A detector that alarms on the
    first day is in cash for the whole window, and in a falling market that
    earns a positive excess return on its own, with no information about when
    the fall arrives. Holding the time in the market fixed and moving only the
    dates isolates the timing. A value near 0.5 means the alarm dates carry
    nothing the calendar did not already give.
    """
    n = len(ret)
    k = int(round(float(pos.sum())))
    if n < 3:
        return np.nan
    if k <= 0 or k >= n:
        # In cash for the whole window, or in the market for the whole window.
        # Either way there is only one arrangement of the days and the dates
        # carry no timing information at all, so the detector sits exactly at
        # the middle of its own null. Dropping these cells instead would
        # quietly remove the detectors that alarm on the first day, which are
        # the ones this test exists to catch.
        return 0.5
    out = np.empty(draws)
    for i in range(draws):
        s = int(rng.integers(0, n - k + 1))
        q = np.zeros(n)
        q[s:s + k] = 1.0
        out[i] = float(np.prod(1.0 + ret * q) - 1.0) * 100.0
    bah = float(np.prod(1.0 + ret) - 1.0) * 100.0
    return float((out - bah < delta).mean())


def positions(alarm, rule, k, m):
    """Whether the asset is held on each decision day."""
    n = len(alarm)
    if rule == "defensive":
        return (alarm == 0).astype(float)
    if rule == "first_alarm":
        pos = np.ones(n)
        hit = np.flatnonzero(alarm == 1)
        if len(hit):
            pos[hit[0]:] = 0.0
        return pos
    if rule == "persist":
        pos = np.ones(n)
        ra = rc = 0
        held = True
        for i in range(n):
            if alarm[i] == 1:
                ra, rc = ra + 1, 0
            else:
                rc, ra = rc + 1, 0
            if held and ra >= k:
                held = False
            elif not held and rc >= m:
                held = True
            pos[i] = 1.0 if held else 0.0
        return pos
    raise ValueError(rule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--far", default="0.05,0.10,0.20")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--headline", default="first_alarm",
                    choices=["defensive", "first_alarm", "persist"])
    ap.add_argument("--headline-mode", default="as reported")
    ap.add_argument("--keep-majority", action="store_true",
                    help="also report the unweighted 5-of-9 majority alarm; "
                         "it is rebuilt and checked either way, but it is "
                         "left out of the tables and figures by default")
    ap.add_argument("--null-draws", type=int, default=400,
                    help="random exit dates per cell for the timing null")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fars = [float(x) for x in a.far.split(",")]
    rng = np.random.default_rng(0)

    cells = {}
    for ev in EVENTS:
        for asset in ASSETS:
            c = read_cell(ev, asset)
            if c is not None and len(c) > 3:
                cells[(ev, asset)] = c
    print("read %d episode-asset cells from the saved alarms" % len(cells))
    print("decision at window_end; return to the next day; daily returns "
          "clipped at %.0f per cent" % (RET_CLIP * 100))
    print()

    # thresholds from the crash-free control, one per asset and detector
    thr = {}
    for far in fars:
        for asset in ASSETS:
            c = cells.get((CONTROL, asset))
            if c is None:
                continue
            for short in [s for s, _ in COMPONENTS] + ["ENST"]:
                col = "score_%s" % short
                if col in c.columns:
                    thr[(far, asset, short)] = float(
                        np.quantile(c[col], 1.0 - far))

    modes = ["as reported"] + ["far%.2f" % f for f in fars]
    rows = []
    for (ev, asset), c in cells.items():
        px, dates = prices(ev, asset)
        idx = c["window_end"].to_numpy(int)
        keep = idx < len(px)
        if keep.sum() < len(idx):
            print("  %s %s: %d of %d endpoints past the end of the price "
                  "series" % (ev, asset, len(idx) - keep.sum(), len(idx)))
        c, idx = c[keep], idx[keep]
        p = px[idx]
        ret = np.clip(p[1:] / p[:-1] - 1.0, -RET_CLIP, RET_CLIP)
        bah = float(np.prod(1.0 + ret) - 1.0) * 100.0
        onset_i = None
        if ev in ONSET:
            w = np.flatnonzero(dates >= pd.Timestamp(ONSET[ev]))
            onset_i = int(w.min()) if len(w) else None

        for mode in modes:
            alarms = {}
            for short, _ in COMPONENTS:
                if mode == "as reported":
                    alarms[short] = c["alarm_%s" % short].to_numpy(int)
                else:
                    far = float(mode[3:])
                    t = thr.get((far, asset, short))
                    alarms[short] = (c["score_%s" % short].to_numpy(float)
                                     >= t).astype(int) if t is not None else None
            if any(v is None for v in alarms.values()):
                continue
            votes = np.sum([alarms[s] for s, _ in COMPONENTS], axis=0)
            alarms["ENS"] = (votes >= MAJORITY).astype(int)
            if "alarm_ENST" in c.columns:
                if mode == "as reported":
                    alarms["ENST"] = c["alarm_ENST"].to_numpy(int)
                else:
                    t = thr.get((float(mode[3:]), asset, "ENST"))
                    if t is not None:
                        alarms["ENST"] = (c["score_ENST"].to_numpy(float)
                                          >= t).astype(int)

            for meth in METHODS:
                if meth not in alarms:
                    continue
                al = alarms[meth]
                hit = np.flatnonzero(al == 1)
                first_i = int(idx[hit[0]]) if len(hit) else None
                for rule in ("defensive", "first_alarm", "persist"):
                    pos = positions(al, rule, a.k, a.m)[:-1]
                    r = float(np.prod(1.0 + ret * pos) - 1.0) * 100.0
                    rows.append({
                        "null_pct": null_percentile(ret, pos, r - bah,
                                                    a.null_draws, rng),
                        "mode": mode, "strategy": rule, "event": ev,
                        "asset": asset, "method": meth,
                        "final_return_pct": r, "bah_pct": bah,
                        "delta_pct": r - bah,
                        "alarm_rate": float(al.mean()),
                        "days_in_market": float(pos.mean()),
                        "n_endpoints": int(len(idx)),
                        "start_date": str(dates.iloc[idx[0]].date()),
                        "end_date": str(dates.iloc[idx[-1]].date()),
                        "first_alarm": (str(dates.iloc[first_i].date())
                                        if first_i is not None else ""),
                        "lead_days": ((onset_i - first_i)
                                      if (first_i is not None
                                          and onset_i is not None)
                                      else np.nan)})

    t = pd.DataFrame(rows)
    # The majority alarm is always rebuilt, because the reference check
    # needs it. It is only reported when asked for.
    check_published(t)
    if not a.keep_majority:
        t = t[t["method"] != "ENS"].copy()
    t.round(6).to_csv(os.path.join(OUT, "backtest_new_cells.csv"), index=False)

    bars = t[(t["mode"] == a.headline_mode)
             & (t["strategy"] == a.headline)].copy()
    bars["method_order"] = bars["method"].map(
        {m: i for i, m in enumerate(METHODS)})
    bars.sort_values(["event", "asset", "method_order"]).round(6).to_csv(
        os.path.join(OUT, "bar_values_new.csv"), index=False)

    write_table(t, a)
    summary(t, a, fars)
    print()
    print("Files written under", OUT)
    print("  backtest_new_cells.csv   every mode, rule and detector")
    print("  bar_values_new.csv       the value behind every bar (%s, %s)"
          % (a.headline, a.headline_mode))
    print("  table_financial_new.tex  the replacement for Table 3")


def check_published(t):
    """Print the defensive rule at the reported thresholds, as a reference."""
    s = t[(t["mode"] == "as reported") & (t["strategy"] == "defensive")]
    piv = s.pivot_table(index=["event", "asset"], columns="method",
                        values="delta_pct")
    bah = s.groupby(["event", "asset"])["bah_pct"].first()
    print("=" * 80)
    print("Defensive rule at the reported thresholds")
    print("=" * 80)
    print("%-18s %-7s %16s %16s %16s"
          % ("episode", "asset", "BaH", "delta EXP", "delta ENS-H"))
    n_ok = [0, 0, 0]
    for k, (pb, pe, pn) in PUBLISHED.items():
        if k not in piv.index:
            print("%-18s %-7s   cell not found" % k)
            continue
        mb, me, mn = (float(bah.loc[k]), float(piv.loc[k, "EXP"]),
                      float(piv.loc[k, "ENS"]))
        for i, (p, m) in enumerate(((pb, mb), (pe, me), (pn, mn))):
            n_ok[i] += abs(m - p) < 0.1
        print("%-18s %-7s %7.1f %8.2f %7.1f %8.2f %7.1f %8.2f"
              % (k[0], k[1], pb, mb, pe, me, pn, mn))
    n = len(PUBLISHED)
    print()
    print("  Buy-and-Hold      %2d of %d cells reproduce" % (n_ok[0], n))
    print("  delta EXP         %2d of %d cells reproduce" % (n_ok[1], n))
    print("  delta ENS-H       %2d of %d cells reproduce" % (n_ok[2], n))
    print()


def write_table(t, a):
    s = t[(t["mode"] == a.headline_mode) & (t["strategy"] == a.headline)]
    piv = s.pivot_table(index=["event", "asset"], columns="method",
                        values="final_return_pct")
    bah = s.groupby(["event", "asset"])["bah_pct"].first()
    rule = {"defensive": "the defensive rule, in cash on any alarm day",
            "first_alarm": "the first-alarm rule: the position is held until "
                           "the first alarm and moved to cash for the "
                           "remainder of the test window",
            "persist": "the persistence rule"}[a.headline]
    lines = [
        r"% Replacement for Table 3 (tab:financial).",
        "%% Rule: %s. Thresholds: %s." % (a.headline, a.headline_mode),
        r"% Produced by backtest_new.py; every value is in "
        r"output/backtest_new_cells.csv.",
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Financial evaluation of core ExpoGAF-AnoNet (EXP) and "
        r"EnsembleExpoGAF (ENS) under " + rule +
        r". Cumulative return against Buy-and-Hold (BaH) over each test "
        r"window, in percent; a positive $\Delta$ means the alarm-driven "
        r"strategy beat holding through. Decisions are taken at the last "
        r"observation of each window and earn the following day's return; "
        r"daily returns are clipped at $\pm 60$\%, which the April 2020 "
        r"negative settlement in WTI requires. Normal 2019 is the crash-free "
        r"July--December control.}",
        r"\label{tab:financial}", r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Episode & Asset & BaH & EXP & $\Delta$EXP & ENS & $\Delta$ENS \\",
        r"\midrule"]

    def num(v):
        if not np.isfinite(v):
            return "--"
        return r"$0.0$" if abs(v) < 0.05 else r"$%+.1f$" % v

    first = True
    for ev in EVENTS:
        here = [x for x in ASSETS if (ev, x) in piv.index]
        if not here:
            continue
        if not first:
            lines.append(r"\addlinespace")
        first = False
        for j, asset in enumerate(here):
            b = float(bah.loc[(ev, asset)])
            e = float(piv.loc[(ev, asset), "EXP"])
            ecol = "ENS" if "ENS" in piv.columns else "ENST"
            n = float(piv.loc[(ev, asset), ecol])
            head = (r"\multirow{%d}{*}{%s} " % (len(here), ev) if j == 0
                    else " ")
            lines.append("%s& %-6s & %s & %s & %s & %s & %s \\\\"
                         % (head, asset, num(b), num(e), num(e - b), num(n),
                            num(n - b)))
    lines += [r"\bottomrule", r"\end{tabular}", r"\normalsize",
              r"\end{table*}"]
    with open(os.path.join(OUT, "table_financial_new.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def summary(t, a, fars):
    from scipy.stats import binomtest
    crisis = t[t["event"] != CONTROL]
    for rule in ("defensive", "first_alarm", "persist"):
        sub = crisis[crisis["strategy"] == rule]
        mean = sub.pivot_table(index="method", columns="mode",
                               values="delta_pct")
        med = sub.pivot_table(index="method", columns="mode",
                              values="delta_pct", aggfunc="median")
        cols = [c for c in ["as reported"] + ["far%.2f" % f for f in fars]
                if c in mean.columns]
        keep = [m for m in METHODS if m in mean.index]
        out = mean[cols].reindex(keep)
        out["median %s" % cols[0]] = med[cols[0]].reindex(keep)
        out.index = [FULL[m] for m in keep]
        print("=" * 80)
        print({"defensive": "Defensive rule: in cash on any alarm day",
               "first_alarm": "First alarm: in cash from the first alarm to "
                              "the end of the window",
               "persist": "Persistence: out after %d alarms, back after %d "
                          "clear days" % (a.k, a.m)}[rule])
        print("=" * 80)
        print("Return above Buy-and-Hold, per cent, over the fifteen crisis "
              "cells")
        print(out.round(2).to_string())
        v = out[cols[0]].astype(float)
        print("  best under %s: %s at %+.2f; ExpoGAF %+.2f, ensemble %+.2f"
              % (cols[0], v.idxmax(), v.max(), v[FULL["EXP"]],
                 v[FULL["ENST"]]))
        print()

    fa = crisis[(crisis["strategy"] == "first_alarm")
                & (crisis["mode"] == a.headline_mode)]
    g = fa.groupby("method")["lead_days"].agg(
        cells="size", median="median",
        early=lambda x: int((x > 0).sum()))
    g = g.reindex([m for m in METHODS if m in g.index])
    g.index = [FULL[m] for m in g.index]
    print("=" * 80)
    print("Trading days between the first alarm and the onset (%s thresholds)"
          % a.headline_mode)
    print("=" * 80)
    print("positive means the alarm came before the onset")
    print(g.round(1).to_string())

    print()
    print("=" * 80)
    print("Timing test: the excess return against random exit dates that")
    print("spend exactly as long in the market (%s rule, %s thresholds)"
          % (a.headline, a.headline_mode))
    print("=" * 80)
    print("0.5 means the alarm dates carry nothing the calendar did not give")
    nl = crisis[(crisis["strategy"] == a.headline)
                & (crisis["mode"] == a.headline_mode)]
    gn = nl.groupby("method").agg(
        cells=("null_pct", "count"),
        in_market=("days_in_market", "mean"),
        beats_random=("null_pct", "mean"),
        above_half=("null_pct", lambda x: int((x > 0.5).sum()))).reindex(
        [m for m in METHODS if m in set(nl["method"])])
    gn.index = [FULL[m] for m in gn.index]
    print(gn.round(3).to_string())
    v = nl.groupby("method")["delta_pct"].mean()
    w = nl.groupby("method")["days_in_market"].mean()
    both = pd.concat([v, w], axis=1).dropna()
    if len(both) > 2:
        r = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
        print()
        print("  correlation between excess return and time in the market "
              "%+.3f" % r)
        if r < -0.5:
            print("  the ranking under this rule is driven by how long each")
            print("  detector sits in cash, not by when it fires; read the")
            print("  column above instead")

    s = crisis[(crisis["strategy"] == a.headline)
               & (crisis["mode"] == a.headline_mode)]
    piv = s.pivot_table(index=["event", "asset"], columns="method",
                        values="delta_pct")
    print()
    print("=" * 80)
    print("EXP and ENS against each baseline under the %s rule, paired over "
          "the fifteen cells, Holm corrected" % a.headline)
    print("=" * 80)
    for ours in ("EXP", "ENS", "ENST"):
        if ours not in piv.columns:
            continue
        res = []
        for m in METHODS:
            if m in ("EXP", "ENS", "ENST") or m not in piv.columns:
                continue
            v = (piv[ours] - piv[m]).dropna()
            w = int((v > 0).sum())
            res.append((m, v.mean(), w, len(v),
                        binomtest(w, len(v), 0.5).pvalue))
        res.sort(key=lambda r: r[4])
        n = len(res)
        print()
        print("  %s" % FULL[ours])
        print("  %-22s %10s %9s %9s %9s" % ("against", "mean diff", "wins",
                                            "p", "Holm p"))
        run_max = 0.0
        surv = 0
        for i, (m, md, w, nn, p) in enumerate(res):
            adj = min(1.0, max(run_max, (n - i) * p))
            run_max = adj
            surv += adj < 0.05
            print("  %-22s %+10.2f %5d/%-3d %9.3f %9.3f"
                  % (FULL[m], md, w, nn, p, adj))
        print("  surviving Holm at 0.05: %d of %d" % (surv, n))


if __name__ == "__main__":
    main()

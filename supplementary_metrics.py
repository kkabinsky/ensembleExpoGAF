# -*- coding: utf-8 -*-
"""
supplementary_metrics.py
========================

Rebuilds Supplementary Tables S1 to S3 as Reviewer 3 point 4 and Reviewer 4
point 3 ask.

Reviewer 3 point 4
    S1 to S3 report neither the number of observations nor the split between
    positive and negative windows, and score performance by accuracy alone,
    which is misleading under class imbalance. Precision, recall, F1, balanced
    accuracy and AUC are requested.

Reviewer 4 point 3
    PR-AUC is requested as well, for the same reason.

Everything is computed from the per-window scores already stored, so no model
is retrained and every figure traces back to the run that produced the
submitted tables.

What to know before reading the output
--------------------------------------
standalone_tadgan is excluded. It was scored on windows of 100 observations
while every other method uses 32. A longer window overlaps the event interval
more readily, so its positive counts differ: COVID-19 EURUSD carries 94 against
74 for every other method. That is a different unit of analysis, not a
labelling error, and the two cannot be compared. Its lead-time figures are
withdrawn on the same grounds. To bring it back, rescore it at window 32.

Run
    python supplementary_metrics.py
"""
import argparse
import glob
import io
import os
import sys

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "Iran_new_run")
OUT = os.path.join(HERE, "output_supplementary")

EPISODES = {
    "COVID-19":            "results_covid",
    "Russia-Ukraine":      "results_russia",
    "Chinese real estate": "results_chinese",
    "Iran 2025":           "results",
    "Iran 2026":           "results_2026",
}
CONTROL = os.path.join(RUN, "covid_normal", "results")
ASSETS = ["USOIL", "GOLD", "EURUSD"]

# excluded: its window length differs from every other method, see above
EXCLUDE = {"standalone_tadgan"}
# The detectors the manuscript uses, pinned here. Without this the script
# would pick up any directory added later under the results tree and the
# output would stop matching the manuscript.
PAPER_METHODS = {
    "anomaly_transformer", "autoencoder", "cnn_autoencoder", "dagmm",
    "deep_svdd", "hybrid2", "isolation_forest", "omnianomaly",
    "one_class_svm", "standalone_fanogan", "tranad", "usad", "vae",
}


# ---------------------------------------------------------------------------
# metrics implemented here rather than taken from sklearn
# ---------------------------------------------------------------------------
def auc_score(y, s):
    """AUC from the Mann-Whitney rank statistic, ties handled."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(ss):                       # average the ranks of ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def pr_auc_score(y, s):
    """Average precision as a step sum, with no interpolation."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    npos = int((y == 1).sum())
    if npos == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    prec = tp / k
    return float((prec * y).sum() / npos)


def threshold_metrics(y, pred):
    """Precision, Recall, F1, Balanced Accuracy, Accuracy, FPR"""
    y = np.asarray(y, int)
    p = np.asarray(pred, int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    f1 = (2 * prec * rec / (prec + rec)
          if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else np.nan)
    bal = (rec + spec) / 2 if np.isfinite(rec) and np.isfinite(spec) else np.nan
    acc = (tp + tn) / len(y) if len(y) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=prec, recall=rec,
                f1=f1, balanced_acc=bal, accuracy=acc, fpr=fpr)


# ---------------------------------------------------------------------------
def read_scores(root, asset, method, mapping):
    p = (os.path.join(root, asset, method, mapping, "test_scores.csv")
         if mapping != "-" else os.path.join(root, asset, method, "test_scores.csv"))
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p).sort_values("window_start")
    return d


def discover(root):
    """Return the (method, mapping) pairs present under the results tree."""
    out = set()
    for asset in ASSETS:
        base = os.path.join(root, asset)
        if not os.path.isdir(base):
            continue
        for m in os.listdir(base):
            mp = os.path.join(base, m)
            if not os.path.isdir(mp):
                continue
            if m not in PAPER_METHODS:
                continue
            if os.path.exists(os.path.join(mp, "test_scores.csv")):
                out.add((m, "-"))
            for g in os.listdir(mp):
                if os.path.exists(os.path.join(mp, g, "test_scores.csv")):
                    out.add((m, g))
    return sorted(out)


def tau_for_far(control_scores, target):
    """Threshold holding the control false-alarm rate at or below target."""
    c = np.asarray(control_scores, float)
    c = c[np.isfinite(c)]
    if len(c) == 0:
        return np.nan, np.nan
    cand = np.unique(c)
    cand = np.concatenate([cand, [cand[-1] + abs(cand[-1]) * 1e-6 + 1e-12]])
    best_t, best_f = float(cand[-1]), float((c >= cand[-1]).mean())
    for t in cand[::-1]:
        f = float((c >= t).mean())
        if f <= target + 1e-12:
            best_t, best_f = float(t), f
        else:
            break
    return best_t, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--far", type=float, default=0.10,
                    help="target false-alarm rate on the control")
    ap.add_argument("--exclude-episode", nargs="*", default=[],
                    help="episodes to leave out entirely")
    ap.add_argument("--no-rank-metrics", nargs="*", default=["Iran 2026"],
                    help="episodes that keep their counts and threshold metrics "
                         "but report no AUC or PR-AUC, being unrankable")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.exclude_episode:
        print("Episodes excluded: %s" % ", ".join(a.exclude_episode))

    rows = []
    skipped = []
    for ep, sub in EPISODES.items():
        if ep in a.exclude_episode:
            continue
        root = os.path.join(RUN, sub)
        if not os.path.isdir(root):
            continue
        for method, mapping in discover(root):
            if method in EXCLUDE:
                skipped.append((ep, method))
                continue
            for asset in ASSETS:
                d = read_scores(root, asset, method, mapping)
                if d is None:
                    continue
                y = d["label_0normal_1crash"].to_numpy(int)
                s = pd.to_numeric(d["anomaly_score"], errors="coerce").to_numpy(float)
                n, npos = len(y), int(y.sum())
                if npos == 0 or npos == n:
                    continue
                rec = {"episode": ep, "asset": asset, "method": method,
                       "mapping": mapping, "n_windows": n,
                       "n_positive": npos, "n_negative": n - npos,
                       "prevalence": round(npos / n, 4),
                       "auc": auc_score(y, s), "pr_auc": pr_auc_score(y, s)}
                rec["pr_auc_random"] = round(npos / n, 4)
                # episodes whose test set ends inside the event cannot be ranked
                # see the note at the top of this file
                rec["rank_metrics_valid"] = ep not in a.no_rank_metrics

                # (a) the threshold shipped with the result file
                if "predicted_label" in d.columns:
                    m1 = threshold_metrics(y, d["predicted_label"].to_numpy(int))
                    for k, v in m1.items():
                        rec["shipped_" + k] = v

                # (b) a threshold holding the false-alarm rate equal across methods
                ct = read_scores(CONTROL, asset, method, mapping)
                if ct is not None:
                    cs = pd.to_numeric(ct["anomaly_score"],
                                       errors="coerce").to_numpy(float)
                    tau, far = tau_for_far(cs, a.far)
                    if np.isfinite(tau):
                        m2 = threshold_metrics(y, (s >= tau).astype(int))
                        rec["far_tau"] = round(float(tau), 6)
                        rec["far_control"] = round(float(far), 4)
                        for k, v in m2.items():
                            rec["matched_" + k] = v
                rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no results; check that the results directories are present")
        return
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(4)
    df.to_csv(os.path.join(OUT, "supplementary_metrics.csv"), index=False)
    print("Wrote supplementary_metrics.csv, %d rows" % len(df))
    if skipped:
        print("Excluded: standalone_tadgan (window 100, not 32), %d times"
              % len(skipped))

    # counts and class balance, the first thing Reviewer 3 asked for
    counts = (df.groupby(["episode", "asset"])
                .agg(n_windows=("n_windows", "max"),
                     n_positive=("n_positive", "max"),
                     n_negative=("n_negative", "max"),
                     prevalence=("prevalence", "max"),
                     n_methods=("method", "nunique"),
                     distinct_pos=("n_positive", "nunique")).reset_index())
    counts.to_csv(os.path.join(OUT, "class_counts.csv"), index=False)
    print()
    print("=" * 88)
    print("Window counts and class balance (distinct_pos must be 1 in every row)")
    print("=" * 88)
    print(counts.to_string(index=False))
    if (counts["distinct_pos"] != 1).any():
        print()
        print("WARNING: some rows disagree on the positive count across methods")

    dv = df[df["rank_metrics_valid"]]
    print()
    print("=" * 88)
    print("Mean AUC and PR-AUC across episodes and assets (threshold-free)")
    if a.no_rank_metrics:
        print("Not included: %s" % ", ".join(a.no_rank_metrics))
    print("=" * 88)
    g = (dv.groupby(["method", "mapping"])
           .agg(auc=("auc", "mean"), pr_auc=("pr_auc", "mean"),
                pr_auc_random=("pr_auc_random", "mean"),
                n=("auc", "count")).reset_index())
    g["pr_auc_lift"] = (g["pr_auc"] - g["pr_auc_random"]).round(4)
    print(g.sort_values("pr_auc", ascending=False).round(4).to_string(index=False))

    print()
    print("=" * 88)
    print("F1 at the shipped threshold against F1 at a matched FAR of %.2f" % a.far)
    print("=" * 88)
    h = (df.groupby(["method", "mapping"])
           .agg(shipped_f1=("shipped_f1", "mean"),
                shipped_fpr=("shipped_fpr", "mean"),
                matched_f1=("matched_f1", "mean"),
                matched_fpr=("matched_fpr", "mean")).reset_index())
    print(h.sort_values("matched_f1", ascending=False).round(4).to_string(index=False))

    write_latex(df, a.far, a.no_rank_metrics)
    print()
    print("Output written to", OUT)


def write_latex(df, far, no_rank):
    """Write LaTeX tables ready to \\input, one per episode.

    Episodes in no_rank keep their counts and threshold metrics but leave the
    AUC and PR-AUC cells as a dash, with the reason stated in the caption.
    That is better than dropping the episode silently, since the reviewer
    asked for the counts of every episode."""
    path = os.path.join(OUT, "supplementary_tables.tex")
    lines = ["% generated by supplementary_metrics.py; do not edit by hand",
             "% every figure comes from the stored per-window scores, nothing retrained",
             "% standalone_tadgan is absent: it was scored at window 100, not 32",
             ""]
    order = [e for e in EPISODES if e in set(df["episode"])]
    for ep in order:
        g = df[df["episode"] == ep]
        skip_rank = ep in no_rank
        cap = (r"%s: detection performance on the event-positive windows. "
               r"$n$ is the number of scored windows and $n_{+}$ the number of "
               r"event-positive windows; the positive-class prevalence is given "
               r"as Prev. Precision, recall and F1 are evaluated with each "
               r"detector's decision threshold calibrated on the crash-free 2019 "
               r"control to a common false-alarm rate of %.2f, so that all "
               r"detectors are compared at the same operating point. AUC and "
               r"PR-AUC require no threshold; the PR-AUC of a random ranking "
               r"equals the prevalence and is stated for reference."
               % (ep.replace("&", r"\&"), far))
        if skip_rank:
            cap += (r" Ranking metrics are not reported for this episode. The "
                    r"data vintage ends inside the event interval, so the "
                    r"event-positive windows form the tail of the test set and "
                    r"no post-event negative region exists against which the "
                    r"scores could be ranked. Threshold metrics remain "
                    r"interpretable and are reported.")
        lines += [
            r"\begin{table}[ht]", r"\centering", r"\small",
            r"\caption{%s}" % cap,
            r"\label{tab:supp_%s}"
            % ep.lower().replace(" ", "_").replace("-", "_"),
            r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
            r"Asset & Method & $n$ & $n_{+}$ & Prev. & AUC & PR-AUC "
            r"& Prec. & Rec. & F1 \\",
            r"\midrule",
        ]

        def fmt(x):
            return "--" if x is None or not np.isfinite(x) else ("%.3f" % x)

        for asset in ASSETS:
            sub = g[g["asset"] == asset]
            key = "matched_f1" if skip_rank else "pr_auc"
            sub = sub.sort_values(key, ascending=False)
            for _, r in sub.iterrows():
                name = r["method"].replace("_", r"\_")
                if r["mapping"] not in ("-", None):
                    name += " (%s)" % r["mapping"]
                auc_c = "--" if skip_rank else fmt(r["auc"])
                pra_c = "--" if skip_rank else fmt(r["pr_auc"])
                lines.append(
                    "%s & %s & %d & %d & %.3f & %s & %s & %s & %s & %s \\\\"
                    % (asset, name, r["n_windows"], r["n_positive"],
                       r["prevalence"], auc_c, pra_c,
                       fmt(r.get("matched_precision", np.nan)),
                       fmt(r.get("matched_recall", np.nan)),
                       fmt(r.get("matched_f1", np.nan))))
            lines.append(r"\midrule")
        if lines[-1] == r"\midrule":
            lines.pop()
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    print("Wrote supplementary_tables.tex (%d tables)" % len(order))


if __name__ == "__main__":
    main()

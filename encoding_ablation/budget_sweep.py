# -*- coding: utf-8 -*-
"""
budget_sweep.py
===============

Does the ordering of the four angular mappings depend on how long the model is
trained?

A short run separated the mappings cleanly and put the exponential map first. A
longer run on the same data put it third. Only one of those can be reported, and
which one is not a matter of preference: an ordering that changes with the
training budget is a statement about the budget.

This program repeats the comparison at several budgets, changing nothing else,
and reports the gap between the leading mapping and the runner-up against the
spread across seeds at the same budget. If the gap shrinks as the budget grows
while the seed spread does not, the clean separation at a short budget was
undertraining rather than a property of the encoding.

The classifier is used because it is the cheapest arm and the one whose short
run produced the clean answer; the same sweep on the adversarial arms costs
hours rather than minutes and can be run with `--arm fanogan_compact`.

Stopping and resuming
    Every finished cell is written immediately. Interrupt it and rerun the same
    command: it continues from the next unfinished cell.

Run
    python budget_sweep.py
    python budget_sweep.py --budgets 3,5,10,20,40 --seeds 3
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import gaf_encodings as enc
import models as mdl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="3,5,10,20,40")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arm", default="classifier",
                    choices=["classifier", "fanogan_compact"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    budgets = [int(x) for x in a.budgets.split(",")]
    device = torch.device("cpu")

    data = enc.build()
    if data is None:
        raise SystemExit("images could not be built; check ../Iran_new_run")

    csv = os.path.join(OUT, "budget_sweep_%s%s.csv" % (a.arm, a.tag))
    rows, seen = [], set()
    if os.path.exists(csv):
        try:
            prev = pd.read_csv(csv)
            rows = prev.to_dict("records")
            seen = {(int(r["epochs"]), str(r["mapping"]), str(r["held_out"]),
                     int(r["seed"])) for _, r in prev.iterrows()}
            print("found %d earlier rows; continuing" % len(seen), flush=True)
        except Exception as e:
            print("could not read earlier results (%s); starting over" % e)

    def flush():
        tmp = csv + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, csv)

    total = len(budgets) * len(enc.ORDER) * len(enc.EPISODES) * a.seeds
    print("arm %s   budgets %s   seeds %d   %d cells in total"
          % (a.arm, budgets, a.seeds, total), flush=True)
    print(flush=True)

    t0 = time.time()
    n = 0
    for ep_budget in budgets:
        for m in enc.ORDER:
            X, y, tags = data[m]["X"], data[m]["y"], data[m]["ep"]
            for held in enc.EPISODES:
                te = tags == held
                tr = ~te
                if te.sum() == 0 or y[te].sum() in (0, int(te.sum())):
                    continue
                for sd in range(a.seeds):
                    n += 1
                    if (ep_budget, m, held, sd) in seen:
                        continue
                    torch.manual_seed(sd)
                    if a.arm == "classifier":
                        model = mdl.make_classifier()
                        s = mdl.train_classifier(model, X[tr], y[tr], X[te],
                                                 ep_budget, sd)
                    else:
                        s, _ = mdl.fit_fanogan(X[tr & (y == 0)], X[te],
                                               "compact", ep_budget, sd,
                                               device)
                    rows.append({"arm": a.arm, "epochs": ep_budget,
                                 "mapping": m, "held_out": held, "seed": sd,
                                 "auc": enc.auc(y[te], s),
                                 "n_test": int(te.sum()),
                                 "prevalence": float(y[te].mean())})
                    flush()
                    print("budget %-3d %-12s %-16s seed %d   AUC %.4f   "
                          "[%d/%d  %.0f min]"
                          % (ep_budget, m, held, sd, rows[-1]["auc"], n, total,
                             (time.time() - t0) / 60), flush=True)

    report(csv)


def report(csv):
    df = pd.read_csv(csv)
    df = df[df["auc"].notna()]
    if df.empty:
        print("no results yet")
        return
    print()
    print("=" * 82)
    print("Mean AUC by budget and mapping")
    print("=" * 82)
    p = df.pivot_table(index="epochs", columns="mapping", values="auc")
    p = p[[c for c in enc.ORDER if c in p.columns]]
    p["best"] = p.idxmax(axis=1)
    p["exp_rank"] = [
        int(p.loc[i, [c for c in enc.ORDER if c in p.columns]].astype(float)
            .rank(ascending=False)["exponential"]) for i in p.index]
    print(p.round(4).to_string())

    print()
    print("=" * 82)
    print("Does the separation survive a longer budget?")
    print("=" * 82)
    print("%-8s %10s %12s %12s %10s"
          % ("budget", "leader", "lead over 2nd", "seed spread", "lead/spread"))
    for b in sorted(df["epochs"].unique()):
        s = df[df["epochs"] == b]
        v = s.groupby("mapping")["auc"].mean().reindex(
            [c for c in enc.ORDER if c in set(s["mapping"])])
        srt = v.sort_values(ascending=False)
        lead = float(srt.iloc[0] - srt.iloc[1])
        spread = float(s.groupby(["mapping", "held_out"])["auc"].std().mean())
        ratio = lead / spread if spread > 0 else float("nan")
        print("%-8d %10s %12.4f %12.4f %10.2f"
              % (b, srt.index[0], lead, spread, ratio))
    print()
    print("  A ratio near or below one means the leader is not separated from")
    print("  the runner-up by more than a change of seed produces at the same")
    print("  budget. Read the trend down the column, not any single row.")

    print()
    print("=" * 82)
    print("Where the exponential mapping ranks at each budget")
    print("=" * 82)
    for b in sorted(df["epochs"].unique()):
        s = df[df["epochs"] == b]
        v = s.groupby("mapping")["auc"].mean()
        r = int(v.rank(ascending=False)["exponential"])
        print("  budget %-4d rank %d of %d   AUC %.4f   leader %s %.4f"
              % (b, r, len(v), v["exponential"], v.idxmax(), v.max()))
    print()
    print("Output file", csv)


if __name__ == "__main__":
    main()

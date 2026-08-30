# -*- coding: utf-8 -*-
"""
lambda_sweep.py
===============

Is the gradient penalty what removes the difference between the four angular
mappings?

The four mappings do not produce images of the same spread: the standard
deviation of the image values is 0.626 for the exponential map, 0.469 for
cosine, 0.421 for arccosh and 0.248 for arctan. Under the supervised classifier
that ordering is reproduced exactly, and under the adversarial one-class model
it is not. The obvious suspect is the gradient penalty, which constrains the
norm of the critic's gradient with respect to its input to one:

    ((grad.norm(2, dim=[1, 2, 3]) - 1) ** 2).mean()

That constraint acts directly on the quantity the mappings differ in. If it is
what erases the difference, then lowering its weight should let the difference
reappear, and the exponential mapping should recover ground.

The test states its prediction before it is run, and one weight is changed and
nothing else. A result in either direction is reportable: if the ordering does
not move when the weight falls, the explanation above is wrong and we say so.

Two weights are compared, 10 as published and 2. Zero is not included on
purpose: removing the penalty altogether drops the Lipschitz constraint that
the Wasserstein objective needs, and what comes back is instability rather than
an answer.

Stopping and resuming
    Every finished cell is written immediately. Interrupt it and rerun the same
    command: it continues from the next unfinished cell.

Run
    python lambda_sweep.py
    python lambda_sweep.py --lambdas 1,2,5,10 --seeds 3
"""
import argparse
import os
import time

import pandas as pd
import torch

import gaf_encodings as enc
import models as mdl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", default="2,10")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--variant", default="compact",
                    choices=["compact", "paper"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    lams = [float(x) for x in a.lambdas.split(",")]
    device = torch.device("cpu")

    data = enc.build()
    if data is None:
        raise SystemExit("images could not be built; check ../Iran_new_run")

    csv = os.path.join(OUT, "lambda_sweep_%s%s.csv" % (a.variant, a.tag))
    rows, seen = [], set()
    if os.path.exists(csv):
        try:
            prev = pd.read_csv(csv)
            rows = prev.to_dict("records")
            seen = {(float(r["lambda_gp"]), str(r["mapping"]),
                     str(r["held_out"]), int(r["seed"]))
                    for _, r in prev.iterrows()}
            print("found %d earlier rows; continuing" % len(seen), flush=True)
        except Exception as e:
            print("could not read earlier results (%s); starting over" % e)

    def flush():
        tmp = csv + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, csv)

    total = len(lams) * len(enc.ORDER) * len(enc.EPISODES) * a.seeds
    prof = enc.mapping_profile().set_index("mapping")["contrast"]
    print("variant %s   epochs %d   seeds %d   weights %s   %d cells"
          % (a.variant, a.epochs, a.seeds, lams, total), flush=True)
    print("image contrast by mapping: " + ", ".join(
        "%s %.3f" % (k, prof[k]) for k in enc.ORDER), flush=True)
    print("prediction: if the penalty is what flattens the mappings, the "
          "exponential", flush=True)
    print("map should rank higher at the lower weight than at 10", flush=True)
    print(flush=True)

    t0 = time.time()
    n = 0
    for lam in lams:
        for m in enc.ORDER:
            X, y, tags = data[m]["X"], data[m]["y"], data[m]["ep"]
            for held in enc.EPISODES:
                te = tags == held
                tr = ~te
                if te.sum() == 0 or y[te].sum() in (0, int(te.sum())):
                    continue
                for sd in range(a.seeds):
                    n += 1
                    if (lam, m, held, sd) in seen:
                        continue
                    s, d = mdl.fit_fanogan(X[tr & (y == 0)], X[te], a.variant,
                                           a.epochs, sd, device,
                                           lambda_gp=lam)
                    rows.append({"lambda_gp": lam, "variant": a.variant,
                                 "mapping": m, "held_out": held, "seed": sd,
                                 "auc": enc.auc(y[te], s),
                                 "prauc": enc.average_precision(y[te], s),
                                 "d_loss_final": d,
                                 "contrast": float(prof[m]),
                                 "n_test": int(te.sum()),
                                 "prevalence": float(y[te].mean()),
                                 "epochs": a.epochs})
                    flush()
                    print("lambda %-5.1f %-12s %-16s seed %d   AUC %.4f   "
                          "[%d/%d  %.0f min]"
                          % (lam, m, held, sd, rows[-1]["auc"], n, total,
                             (time.time() - t0) / 60), flush=True)

    report(csv)


def report(csv):
    import numpy as np
    from scipy.stats import spearmanr
    df = pd.read_csv(csv)
    df = df[df["auc"].notna()]
    if df.empty:
        print("no results yet")
        return
    print()
    print("=" * 82)
    print("Mean AUC by gradient penalty weight and mapping")
    print("=" * 82)
    p = df.pivot_table(index="lambda_gp", columns="mapping", values="auc")
    p = p[[c for c in enc.ORDER if c in p.columns]]
    p["best"] = p.idxmax(axis=1)
    p["exp_rank"] = [
        int(p.loc[i, [c for c in enc.ORDER if c in p.columns]].astype(float)
            .rank(ascending=False)["exponential"]) for i in p.index]
    print(p.round(4).to_string())

    prof = enc.mapping_profile().set_index("mapping")["contrast"]
    print()
    print("=" * 82)
    print("Does the ordering follow the image contrast at each weight?")
    print("=" * 82)
    print("%-10s %10s %12s %14s %12s"
          % ("lambda", "exp rank", "lead over 2nd", "spearman with",
             "seed spread"))
    print("%-10s %10s %12s %14s %12s" % ("", "", "", "contrast", ""))
    for lam in sorted(df["lambda_gp"].unique()):
        s = df[df["lambda_gp"] == lam]
        v = s.groupby("mapping")["auc"].mean().reindex(
            [c for c in enc.ORDER if c in set(s["mapping"])])
        srt = v.sort_values(ascending=False)
        lead = float(srt.iloc[0] - srt.iloc[1])
        rho, _ = spearmanr(prof.reindex(v.index).to_numpy(float),
                           v.to_numpy(float))
        spread = float(s.groupby(["mapping", "held_out"])["auc"].std().mean())
        print("%-10.1f %10d %12.4f %14.3f %12.4f"
              % (lam, int(v.rank(ascending=False)["exponential"]), lead, rho,
                 spread))

    print()
    print("How to read this")
    print("  The prediction was that lowering the weight lets the mappings")
    print("  separate again and moves the exponential map up. Compare the")
    print("  exponential rank and the correlation with contrast between the")
    print("  two rows, and compare the lead against the seed spread in the")
    print("  same row before calling any of it a difference.")
    print()
    print("Output file", csv)


if __name__ == "__main__":
    main()

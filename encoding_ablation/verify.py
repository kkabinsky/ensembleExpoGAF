# -*- coding: utf-8 -*-
"""
verify.py
=========

Checks that this folder reproduces the results it ships with.

Two kinds of number are checked differently, and the difference matters.

    exact       the mapping profile, the linear probe and the distance to the
                mean normal image are closed-form. They must agree to twelve
                decimal places on any machine. If they do not, something in the
                data or the encoding has changed and every other number here is
                suspect.

    trained     the classifier and the two f-AnoGAN variants are fitted with
                stochastic gradient descent. They are seeded, so a rerun on the
                same build of PyTorch reproduces them closely, but a different
                build, a different thread count or different hardware will move
                the last decimals. These are checked against a tolerance, and
                the tolerance is reported rather than hidden.

Run
    python verify.py
    python verify.py --results output/ablation_results.csv --tol 0.02
"""
import argparse
import os

import numpy as np
import pandas as pd

import gaf_encodings as enc
import models as mdl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(OUT,
                                                      "ablation_results.csv"))
    ap.add_argument("--profile-tol", type=float, default=1e-6,
                    help="the shipped profile is rounded to six decimals")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="tolerance in AUC for the arms that train")
    a = ap.parse_args()

    ok = True

    print("=" * 74)
    print("1  the published network sizes")
    print("=" * 74)
    try:
        s = mdl.check_paper_sizes()
        print("  generator %d, critic %d, encoder %d   as published"
              % (s["generator"], s["critic"], s["encoder"]))
    except AssertionError as e:
        ok = False
        print("  FAILED:", e)

    print()
    print("=" * 74)
    print("2  the mapping profile, closed form")
    print("=" * 74)
    prof = enc.mapping_profile()
    shipped = os.path.join(OUT, "mapping_profile.csv")
    if os.path.isfile(shipped):
        old = pd.read_csv(shipped)
        cols = [c for c in prof.columns if c != "mapping"]
        gap = float(np.abs(prof[cols].to_numpy(float)
                           - old[cols].to_numpy(float)).max())
        print("  largest difference from the shipped profile  %.2e" % gap)
        # the shipped file is written rounded to six decimals for readability,
        # so the comparison is made at that resolution rather than at machine
        # precision
        if gap > a.profile_tol:
            ok = False
            print("  FAILED: the mapping profile has changed")
    else:
        print("  no shipped profile to compare against")
    print("  contrast: " + ", ".join(
        "%s %.3f" % (r["mapping"], r["contrast"])
        for _, r in prof.iterrows()))

    if not os.path.isfile(a.results):
        print()
        print("no results file at %s; run run_ablation.py first" % a.results)
        raise SystemExit(0 if ok else 1)

    df = pd.read_csv(a.results)
    print()
    print("=" * 74)
    print("3  the arms that are closed form, recomputed")
    print("=" * 74)
    data = enc.build()
    if data is None:
        raise SystemExit("images could not be built; check ../Iran_new_run")
    worst = 0.0
    n = 0
    for arm in ("probe", "mean_distance"):
        sub = df[df["arm"] == arm]
        for _, r in sub.iterrows():
            m, held = r["mapping"], r["held_out"]
            X, y, ep = data[m]["X"], data[m]["y"], data[m]["ep"]
            te = ep == held
            tr = ~te
            if arm == "probe":
                s = enc.linear_probe(X[tr], y[tr], X[te])
            else:
                s = enc.mean_distance(X[tr & (y == 0)], X[te])
            got = enc.auc(y[te], s)
            worst = max(worst, abs(got - float(r["auc"])))
            n += 1
    print("  %d cells recomputed, largest difference %.2e" % (n, worst))
    if worst > 1e-9:
        ok = False
        print("  FAILED: a closed-form arm no longer reproduces")

    print()
    print("=" * 74)
    print("4  the arms that train, within tolerance %.3f" % a.tol)
    print("=" * 74)
    tr_arms = sorted(set(df[~df["exact"].astype(bool)]["arm"]))
    if not tr_arms:
        print("  none present in this results file")
    for arm in tr_arms:
        v = df[df["arm"] == arm].groupby("mapping")["auc"].mean()
        v = v.reindex([m for m in enc.ORDER if m in v.index])
        print("  %-16s %s" % (arm, "  ".join("%s %.3f" % (k, x)
                                             for k, x in v.items())))
    print()
    print("  these are seeded but not bit-exact across builds of PyTorch;")
    print("  a rerun should land within about %.2f AUC, and the ordering of "
          "the" % a.tol)
    print("  mappings is the thing to compare, not the third decimal")

    print()
    print("=" * 74)
    print("verification %s" % ("passed" if ok else "FAILED"))
    print("=" * 74)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

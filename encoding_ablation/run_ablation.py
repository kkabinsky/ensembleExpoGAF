# -*- coding: utf-8 -*-
"""
run_ablation.py
===============

Does the ordering of the four angular mappings depend on the reader?

The window sweep of the manuscript compares the mappings with a supervised
classifier and puts the exponential map first. The detector the framework
proposes is one-class and never sees a crash label. This program reads the same
images six ways on the same split and reports where the ordering holds and
where it does not.

    contrast          what each mapping does to the image before any model
                      sees it; no data, no fitting
    probe             ridge least squares on the flattened image, fitted on the
                      training episodes' labels; the discriminative reading in
                      its smallest form, with no seed and no training noise
    mean_distance     distance to the mean of the normal training images; the
                      one-class reading in its smallest form, and the baseline
                      any f-AnoGAN has to beat
    classifier        the supervised convolutional network of the window sweep
    fanogan_paper     the f-AnoGAN of the manuscript, on normal windows only
    fanogan_compact   the same objective with the networks resized to the data
                      and every normalisation layer removed
    pseudo            the compact f-AnoGAN scores the training windows, the
                      quantile the manuscript uses as its alarm threshold turns
                      those scores into labels, and the classifier is fitted on
                      those; no crash label is used anywhere in training

The split is leave-one-episode-out. Windows inside an episode overlap by 31 of
their 32 observations, so a random split would leak almost completely. The
four mappings see identical window positions, identical labels and identical
folds, so the comparison is paired cell by cell.

Start small. The default is five epochs, which is enough to see whether an arm
behaves sensibly and takes a few minutes; raise it once the shape of the result
is clear.

Run
    python run_ablation.py                          five epochs, one seed
    python run_ablation.py --epochs 20 --seeds 3    the full setting
    python run_ablation.py --arms probe,mean_distance,contrast   exact only

Stopping it part way is safe: rerun the same command and it continues from the
rows already in the output file.
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

EXACT = ("probe", "mean_distance")
TRAINED = ("classifier", "fanogan_paper", "fanogan_compact", "pseudo")
ALL_ARMS = EXACT + TRAINED


def score_arm(arm, X, y, tr, te, seed, a, device):
    """One arm on one fold. Returns the scores and a note."""
    if arm == "probe":
        return enc.linear_probe(X[tr], y[tr], X[te]), {}
    if arm == "mean_distance":
        return enc.mean_distance(X[tr & (y == 0)], X[te]), {}
    if arm == "classifier":
        torch.manual_seed(seed)
        m = mdl.make_classifier()
        return mdl.train_classifier(m, X[tr], y[tr], X[te], a.cls_epochs,
                                    seed), {}
    if arm in ("fanogan_paper", "fanogan_compact"):
        variant = "paper" if arm.endswith("paper") else "compact"
        s, d = mdl.fit_fanogan(X[tr & (y == 0)], X[te], variant, a.epochs,
                               seed, device, a.verbose)
        return s, {"d_loss_final": d}
    if arm == "pseudo":
        # the one-class stage sees the training windows and nothing else
        s_tr, d = mdl.fit_fanogan(X[tr & (y == 0)], X[tr], "compact",
                                  a.epochs, seed, device, a.verbose)
        cut = float(np.quantile(s_tr, a.threshold_q))
        pseudo = (s_tr > cut).astype(int)
        if pseudo.sum() < 2 or pseudo.sum() >= len(pseudo) - 1:
            return s_tr[:int(te.sum())], {"pseudo_rate": float(pseudo.mean()),
                                          "d_loss_final": d}
        torch.manual_seed(seed)
        m = mdl.make_classifier()
        out = mdl.train_classifier(m, X[tr], pseudo, X[te], a.cls_epochs, seed)
        agree = float((pseudo == y[tr]).mean())
        return out, {"pseudo_rate": float(pseudo.mean()),
                     "pseudo_agrees_with_truth": agree,
                     "d_loss_final": d}
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--epochs", type=int, default=5,
                    help="epochs for the adversarial stage; start small")
    ap.add_argument("--cls-epochs", type=int, default=20,
                    help="epochs for the classifier, which is tiny and cheap")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--threshold-q", type=float, default=0.95,
                    help="the quantile of the training scores that becomes the "
                         "pseudo-label cut, as in the manuscript")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    os.makedirs(OUT, exist_ok=True)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    prof = enc.mapping_profile()
    prof.round(6).to_csv(os.path.join(OUT, "mapping_profile.csv"), index=False)
    print("=" * 78)
    print("What each mapping does to the image, before any model sees it")
    print("=" * 78)
    p = prof.set_index("mapping")
    p.index.name = None
    print(p.round(3).to_string())
    print()
    print("  the exponential map is the only one whose angular resolution")
    print("  rises with the value it encodes (dphi_low to dphi_high), and it")
    print("  produces the highest image contrast of the four")
    print()

    sizes = mdl.check_paper_sizes()
    print("published f-AnoGAN  generator %d  critic %d  encoder %d"
          % (sizes["generator"], sizes["critic"], sizes["encoder"]))
    print("compact f-AnoGAN    generator %d  critic %d  encoder %d"
          % (mdl.count_parameters(mdl.CompactGenerator()),
             mdl.count_parameters(mdl.CompactCritic()),
             mdl.count_parameters(mdl.CompactEncoder())))
    print("classifier          %d"
          % mdl.count_parameters(mdl.make_classifier()))
    print()

    data = enc.build()
    if data is None:
        raise SystemExit("images could not be built; check ../Iran_new_run")
    n = len(data[enc.ORDER[0]]["y"])
    print("%d windows per mapping, positive rate %.3f, %d episodes"
          % (n, data[enc.ORDER[0]]["y"].mean(), len(enc.EPISODES)))
    print("epochs %d, classifier epochs %d, seeds %d"
          % (a.epochs, a.cls_epochs, a.seeds))
    print()

    csv = os.path.join(OUT, "ablation_results%s.csv" % a.tag)
    rows, seen = [], set()
    if os.path.exists(csv):
        try:
            prev = pd.read_csv(csv)
            rows = prev.to_dict("records")
            seen = {(str(r["arm"]), str(r["mapping"]), str(r["held_out"]),
                     int(r["seed"])) for _, r in prev.iterrows()}
            print("found %d earlier rows; continuing" % len(seen))
        except Exception as e:
            print("could not read earlier results (%s); starting over" % e)

    def flush():
        tmp = csv + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, csv)

    t0 = time.time()
    total = len(arms) * len(enc.ORDER) * len(enc.EPISODES) * a.seeds
    done = 0
    for arm in arms:
        seeds = [0] if arm in EXACT else list(range(a.seeds))
        for m in enc.ORDER:
            X, y, ep = data[m]["X"], data[m]["y"], data[m]["ep"]
            for held in enc.EPISODES:
                te = ep == held
                tr = ~te
                if te.sum() == 0 or y[te].sum() in (0, int(te.sum())):
                    continue
                for sd in seeds:
                    done += 1
                    if (arm, m, held, sd) in seen:
                        continue
                    s, extra = score_arm(arm, X, y, tr, te, sd, a, device)
                    row = {"arm": arm, "mapping": m, "held_out": held,
                           "seed": sd, "auc": enc.auc(y[te], s),
                           "prauc": enc.average_precision(y[te], s),
                           "n_train": int(tr.sum()),
                           "n_train_normal": int((tr & (y == 0)).sum()),
                           "n_test": int(te.sum()),
                           "prevalence": float(y[te].mean()),
                           "n_eff": int(te.sum()) / float(enc.W_REF),
                           "epochs": a.epochs, "cls_epochs": a.cls_epochs,
                           "exact": arm in EXACT}
                    row.update(extra)
                    rows.append(row)
                    flush()
                    print("%-16s %-12s %-16s seed %d   AUC %.4f   "
                          "[%d/%d  %.0f min]"
                          % (arm, m, held, sd, row["auc"], done, total,
                             (time.time() - t0) / 60), flush=True)

    report(csv)


def report(csv):
    df = pd.read_csv(csv)
    df = df[df["auc"].notna()]
    if df.empty:
        print("no results yet")
        return
    print()
    print("=" * 78)
    print("Mean AUC by arm and mapping, over the held-out episodes and seeds")
    print("=" * 78)
    p = df.pivot_table(index="arm", columns="mapping", values="auc")
    p = p[[c for c in enc.ORDER if c in p.columns]]
    p["best"] = p.idxmax(axis=1)
    p["exp_rank"] = [int(p.loc[i, [c for c in enc.ORDER if c in p.columns]]
                         .astype(float).rank(ascending=False)["exponential"])
                     for i in p.index]
    order = [x for x in ALL_ARMS if x in p.index]
    print(p.reindex(order).round(4).to_string())

    print()
    print("=" * 78)
    print("Does the ordering agree between the readings?")
    print("=" * 78)
    for arm in order:
        r = p.loc[arm, [c for c in enc.ORDER if c in p.columns]].astype(float)
        print("  %-16s best %-12s exponential ranks %d of %d at %.4f"
              % (arm, r.idxmax(), int(r.rank(ascending=False)["exponential"]),
                 len(r), r["exponential"]))

    prof = enc.mapping_profile().set_index("mapping")["contrast"]
    print()
    print("  correlation with image contrast, across the four mappings")
    for arm in order:
        r = p.loc[arm, [c for c in enc.ORDER if c in p.columns]].astype(float)
        c = prof.reindex(r.index).to_numpy(float)
        print("    %-16s %+.3f" % (arm, np.corrcoef(c, r.to_numpy(float))[0, 1]))

    tr = df[~df["exact"]]
    if not tr.empty:
        print()
        print("=" * 78)
        print("Seed spread, which any difference between mappings has to beat")
        print("=" * 78)
        g = tr.groupby(["arm", "mapping", "held_out"])["auc"].std()
        for arm in [x for x in TRAINED if x in set(tr["arm"])]:
            v = g.loc[arm].dropna()
            if len(v):
                print("  %-16s mean standard deviation across seeds %.4f"
                      % (arm, float(v.mean())))

    if "pseudo_agrees_with_truth" in df.columns:
        s = df[df["arm"] == "pseudo"]["pseudo_agrees_with_truth"].dropna()
        if len(s):
            print()
            print("  the pseudo-labels agree with the crash labels on %.1f%% "
                  "of training windows" % (100 * s.mean()))

    print()
    print("Output file", csv)


if __name__ == "__main__":
    main()

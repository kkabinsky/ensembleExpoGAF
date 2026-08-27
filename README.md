# ExpoGAF-AnoNet — code and data for the revised submission

Code and result files for:

> **ExpoGAF-AnoNet: A Layered Generative Framework with Exponential Gramian
> Angular Encoding for Detecting Anomalous Market Disorders During Major Crash
> Periods**
> Kabin Kanjamapornkul and Theepakorn Jithitikulchai
> Faculty of Economics, Thammasat University
> *Iran Journal of Computer Science*, submission
> `f28943da-999b-4682-92be-d4c1f9b78031`, second revision.

Everything needed to reproduce every number in the manuscript: the model code,
the analysis code, the per-window scores, and the price series. Clone it and the
analysis programs run without editing a path.

## Layout

```
Iran_new_run/
  improved_tadgans_anomaly2.py    TadGAN
  fanogan_four_gaf_compare.py     the four GAF mappings and f-AnoGAN
  oil_baselines_gaf.py            the eight baseline detectors
  run_all_datebased.py            the pipeline as submitted
  run_hybrid_true.py              the corrected pipeline
  ensembleExpoGAF/                the ensemble decision layer and its data
  datasets/, datasets_2026/       daily prices for USOIL, GOLD, EURUSD
  results*/                       per-window scores, one folder per episode
  covid_normal/results/           the crash-free 2019 control
  run_pr_auc/, test_ablation/     analysis programs that live beside the data
test_cnn/                         the window and mapping sweeps
outputs/                          the result tables the manuscript quotes
*.py at the root                  the remaining analysis programs
```

## Running

```
pip install numpy pandas scipy scikit-learn matplotlib openpyxl torch
python supplementary_metrics.py
python Iran_new_run/run_pr_auc/pr_auc_all_datasets_new.py
```

Python 3.10 or newer. Set `ALLOW_CPU=1` if no GPU is present. Each program
writes its tables next to itself; compare them with the reference copies in
`outputs/`.

`RUNBOOK.md` lists, for every program, what it reads, what it writes and the
figures it should print. `RESULTS_MAP.md` maps each program to the table it
produces in the manuscript. `outputs/results_tables_v15.xlsx` holds the same
tables in one workbook, with an INDEX sheet naming where each appears.

## Reproducibility

Eight of the eleven analysis programs read stored scores and are exactly
reproducible: run them from a clean clone and every output file matches the copy
in `outputs/` cell for cell. This was checked before deposit and all eight agree,
as do the 88 figures the manuscript and the response letters quote.

Three programs retrain a network — `cnn_window_sweep.py`,
`window_size_check.py` and `dm_test_mappings.py` — and will differ in the last
decimals on different hardware. The copies in `outputs/` are the runs the
manuscript quotes.

**Do not edit the `PAPER_METHODS` set** in `supplementary_metrics.py`,
`brier_decomposition.py` or `leadtime_far_matched.py`. Those programs used to
discover detectors by listing directories, which meant the tables changed when
any new run was added: the detector count went from thirteen to fifteen and the
event-positive count rose by one in twelve of the fifteen cells. The pinned set
is what keeps the output matching the manuscript.

## What the runs establish

Three corrections to the submitted version, each of which changed a conclusion:

- Diebold-Mariano recomputed with a Newey-West estimator at lag 31, because
  adjacent windows share 31 of their 32 observations. Comparisons surviving Holm
  correction fall from 26 of 45 to 3, all three involving the ensemble layer.
- Lead time recomputed with every detector calibrated to a common false-alarm
  rate on a crash-free control. No detector is significantly earlier than any
  other at any of nine operating points.
- The pipeline rebuilt so that the layered and standalone arms are different
  models; in the submitted runs they carried identical scores in all 72 cells.

Two component ablations were added, and both are unfavourable to the framework:
the TadGAN front end used alone reaches mean AUC 0.560 against 0.528 for the
full pipeline over the same fifteen cells, and the core encoding's mean PR-AUC
is 0.457 against 0.470 for a random ordering of the same windows.

## Licence

Code under MIT. Result files and data tables under CC BY 4.0.

# ExpoGAF-AnoNet — code and data for the revised submission

Code and result files for:

> **ExpoGAF-AnoNet: A Layered Generative Framework with Exponential Gramian
> Angular Encoding for Detecting Anomalous Market Disorders During Major Crash
> Periods**
> Kabin Kanjamapornkul and Theepakorn Jithitikulchai
> Faculty of Economics, Thammasat University
> *Iran Journal of Computer Science*, submission
> `f28943da-999b-4682-92be-d4c1f9b78031`.

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
sensitivity_runs/                 threshold, seed, overlap, bootstrap and DM
                                  correction; each writes to output/
encoding_ablation/                the six-arm encoding ablation, the budget and
                                  lambda sweeps, and the mapping geometry and
                                  its closed-form derivation
backtest_new/                     the financial evaluation on the corrected
                                  alarm-to-price alignment, and its figures
outputs/                          the result tables the manuscript quotes
*.py at the root                  the remaining analysis programs
```

`RESULTS_MAP.md` lists every program against the table or figure it produces in
the revised manuscript. Start there if you are checking a specific number.

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

## What the programs compute

The Diebold-Mariano statistics use a Newey-West long-run standard error at lag 31,
because adjacent windows share 31 of their 32 observations, and the Holm step-down
adjustment across all 45 pairs. Lead time is compared with every detector calibrated
to a common false-alarm rate on a crash-free control. The layered and standalone
arms are trained separately so that the two are different models.

## Licence

Code under MIT. Result files and data tables under CC BY 4.0.

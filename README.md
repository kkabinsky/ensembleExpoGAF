# ExpoGAF-AnoNet

ExpoGAF-AnoNet: A Layered Generative Framework with Exponential Gramian Angular
Encoding for Detecting Anomalous Market Disorders During Major Crash Periods.

The model code, the analysis code, the per-window scores and the price series.
Clone it and the analysis programs run without editing a path.

## Layout

```
Iran_new_run/
  improved_tadgans_anomaly2.py    TadGAN
  fanogan_four_gaf_compare.py     the four GAF mappings and f-AnoGAN
  oil_baselines_gaf.py            the eight baseline detectors
  run_all_datebased.py            image built from the raw-price window
  run_hybrid_true.py              image built from the TadGAN score window
  ensembleExpoGAF/                the ensemble decision layer and its data
  datasets/, datasets_2026/       daily prices for USOIL, GOLD, EURUSD
  results*/                       per-window scores, one folder per episode
  covid_normal/results/           the crash-free 2019 control
  run_pr_auc/, test_ablation/     analysis programs beside the data
test_cnn/                         window and mapping sweeps
sensitivity_runs/                 threshold, seed, overlap, bootstrap, DM
encoding_ablation/                encoding ablation, budget and lambda sweeps,
                                  mapping geometry and its derivation
backtest_new/                     financial evaluation and its figures
outputs/                          result tables
*.py at the root                  the remaining analysis programs
```

## Running

```
pip install numpy pandas scipy scikit-learn matplotlib openpyxl torch
python supplementary_metrics.py
python Iran_new_run/run_pr_auc/pr_auc_all_datasets_new.py
```

Python 3.10 or newer. Set `ALLOW_CPU=1` if no GPU is present. Each program
states in its own header what it reads, what it writes and how to run it.

Programs that read stored scores are exactly reproducible: run them from a
clean clone and the output matches the copy in `outputs/`.
`cnn_window_sweep.py`, `window_size_check.py` and `dm_test_mappings.py` retrain
a network and will differ in the last decimals on different hardware.

`outputs/results_tables_v15.xlsx` holds the result tables in one workbook, with
an INDEX sheet.

Do not edit the `PAPER_METHODS` set in `supplementary_metrics.py`,
`brier_decomposition.py` or `leadtime_far_matched.py`. The detector list and the
event-positive counts depend on it.

## Licence

Code under MIT. Result files and data tables under CC BY 4.0.

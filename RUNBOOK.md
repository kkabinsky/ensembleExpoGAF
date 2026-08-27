# Runbook — what each program reads, writes, and produces

Eleven programs. Eight read stored per-window scores and are exactly
reproducible: run them twice on any machine and the output files are identical
to the byte. Three retrain a network and are reproducible only up to training
noise; they are marked below and their shipped outputs are the ones the
manuscript quotes.

## Input data

Two kinds of input, both produced by the main experiment runners:

| Path | What it holds |
|---|---|
| `results_covid/`, `results_russia/`, `results_chinese/`, `results/`, `results_2026/` | one directory per crash episode; inside, one directory per asset, then one per detector, each holding `test_scores.csv` with `window_start`, `window_end`, `label_0normal_1crash`, `anomaly_score`, `predicted_label` |
| `covid_normal/results/` | the crash-free 2019 control, same layout |
| `<episode>/<asset>/tadgan_stage/out/full_scores.csv` | the cached TadGAN per-step anomaly score |
| `datasets/`, `datasets_2026/` | `USOIL_daily_final2.xlsx`, `GOLD_daily_final2.xlsx`, `EURUSD_daily_final2.xlsx`, daily closing prices |
| `ensembleExpoGAF/data/aligned_probabilities_9methods.csv` | 2,092 windows aligned across nine detectors, with `event`, `asset`, `window_start`, `label` |
| `ensembleExpoGAF/data/aligned_hard_predictions_10methods.csv` | the same 2,092 windows with the ensemble column `prob_ENS` |

Episodes are COVID-19, Russia–Ukraine, Chinese real estate, Iran 2025 and Iran
2026; assets are USOIL, GOLD and EUR/USD. Fifteen episode–asset cells carry
positive windows; the 2019 control carries none, so it is excluded wherever
average precision or AUC is computed.

## A note on the pinned method list

Three programs used to discover detectors by listing directories. After the
corrected pipeline was run, two new directories appeared and those programs
silently picked them up: the tables grew from thirteen detectors to fifteen and
the event-positive count rose by one in twelve of the fifteen cells, so
COVID-19 USOIL read 74 positives instead of the 73 the manuscript reports.
`supplementary_metrics.py`, `brier_decomposition.py` and
`leadtime_far_matched.py` now carry an explicit `PAPER_METHODS` set and ignore
anything outside it. Do not edit that set: it is what makes the output match the
manuscript.

## The programs

### 1. `pr_auc_all_datasets_new.py` — exactly reproducible, about 3 seconds

Reads the same two aligned files.
Writes `pr_auc_all_cells_new.csv` (15 rows), `pr_auc_summary_by_method_new.csv`
(10 rows), `pr_auc_ens_where_best_new.csv` (15 rows).

Average precision for ten methods on the fifteen cells, with the positive
prevalence beside each, since the prevalence is the PR-AUC a random ordering
obtains.

Expect mean PR-AUC: One-Class SVM 0.562, Anomaly Transformer 0.555, Isolation
Forest 0.540, OmniAnomaly 0.526, DAGMM 0.522, EnsembleExpoGAF 0.519, TranAD
0.506, ExpoGAF core 0.457, USAD 0.449, Deep SVDD 0.440, against a
random-ordering reference of 0.470.

### 2. `standalone_tadgan_w32.py` — exactly reproducible, about 6 seconds

Reads `<episode>/<asset>/tadgan_stage/out/full_scores.csv` and the price
workbooks.
Writes `standalone_tadgan_w32_cells.csv`, `standalone_tadgan_w32_summary.csv`.

The component ablation. Reduces the 32 cached TadGAN scores inside each window
to one number, by mean and by maximum, using the same window positions, labels
and split as every other arm. Includes a control that scores a window by its
position in the test period and nothing else.

Expect mean AUC over the fifteen cells: full pipeline 0.528, TadGAN maximum
0.560, TadGAN mean 0.520, position control 0.428.

### 3. `supplementary_metrics.py` — exactly reproducible, about 2 seconds

Reads every `test_scores.csv` under the episode directories, restricted to
`PAPER_METHODS`.
Writes `supplementary_metrics.csv` (285 rows), `class_counts.csv` (15 rows).

Rebuilds Supplementary Tables S1 to S3 with the window counts, the class
balance, and the full metric set including PR-AUC.

Expect prevalence between 0.377 and 0.520 across the fifteen cells; COVID-19
USOIL 146 windows of which 73 positive.

### 4. `leadtime_far_matched.py` — exactly reproducible, about 5 seconds

Reads the episode directories, the 2019 control, and the price workbooks.
Writes `leadtime_far_matched.csv` (2,430 rows), `paired_tests.csv` (171 rows),
`component_ablation.csv`.

Sets each detector's threshold on the crash-free control so that all sit at the
same false-alarm rate, then reads lead time. Three false-alarm levels crossed
with three alarm-persistence settings gives nine operating points.

Expect 171 paired comparisons and none surviving Holm correction. On the
control the uncalibrated false-alarm rates run from 0.034 to 0.893.

### 5. `component_ablation.py` — exactly reproducible, about 1 second

Reads the episode directories.
Writes `effective_sample_size.csv`, `ablation_tadgan.csv`.

Expect 1,589 scored windows corresponding to roughly 50 independent
observations under the overlap ratio, and a median lag-1 score autocorrelation
of 0.881.

### 6. `brier_decomposition.py` — exactly reproducible, about 2 seconds

Reads the episode directories, restricted to `PAPER_METHODS`.
Writes `brier_decomposition.csv`, 684 rows.

Murphy decomposition of the Brier evaluation. Expect every detector to score
worse than a constant forecast at the base rate of 0.2461, which is why the
Brier evaluation is withdrawn in this revision.

### 7. `mapping_on_fixed_pipeline.py` — exactly reproducible, about 1 second

Reads the `hybrid_true` and `standalone_ref` arms.
Prints to the console; the table is `hybrid_true_vs_standalone.csv`.

Expect mean AUC over fifteen cells: GAF of TadGAN scores 0.5283, GAF of raw
prices 0.3558, difference +0.1725, the score path winning 9 of 15 cells.

### 8. `mapping_significance.py` — exactly reproducible, about 1 second

Reads the same arms. Prints to the console.

Expect: exponential against arctan, Diebold–Mariano +2.58 with Holm-adjusted
p = 0.029 and a sign test of 13–2 at p = 0.007; against cosine p = 0.451 and
against arccosh p = 0.392; Friedman across the four mappings
chi-squared 7.16, p = 0.067.

### 9. `cnn_window_sweep.py` — retrains, hours

Reads the price workbooks and the `hybrid2` window positions.
Writes `cnn_window_sweep.csv`, 144 rows.

Three window lengths by four mappings by four held-out episodes by three seeds.
Table 4, upper panel. Expect mean AUC at window 32: cosine 0.5944, arctan
0.4988, arccosh 0.5089, exponential 0.6044, with a positive rate of 0.466 and
an effective sample size of 12.4.

Because it trains, a rerun on different hardware will not reproduce these to the
fourth decimal. The shipped `cnn_window_sweep.csv` is the run the manuscript
quotes.

### 10. `window_size_check.py` — retrains f-AnoGAN, about 2.5 hours

Reads the episode directories and the price workbooks.
Writes `window_sweep.log`.

The same sweep under the one-class scorer, exponential mapping only, three
episodes by three assets. Table 4, lower panel. Expect mean AUC 0.4903 at
window 16, 0.4937 at 32 and 0.6572 at 64, with effective sample sizes of 7.9,
4.0 and 2.0.

Same caveat: training noise means a rerun will differ in the last decimals.

### 11. `dm_test_mappings.py` — retrains unless the cache is present, minutes

Reads `gaf_dataset.npz` if present, otherwise rebuilds it from the episode
directories.
Writes `dm_cnn_mappings.csv`, `dm_fanogan_mappings.csv`, and the per-episode
versions.

Expect, under the supervised classifier, exponential better than arccosh with
DM 2.864 and Holm-adjusted p = 0.025, and no pair surviving under f-AnoGAN.

## Checking a rerun

The eight deterministic programs were rerun from a clean state and every output
file matched the shipped copy cell for cell. If a rerun disagrees, check first
that `PAPER_METHODS` has not been edited and that no extra detector directory
has been added under the episode folders.

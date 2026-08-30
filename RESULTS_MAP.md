# Which program produced which table

Table and figure numbers are those of the revised manuscript. Every program
below runs from inside the folder it lives in and reads only files that ship
with this repository.

## Main-text tables and figures

| Program | Result files | Manuscript |
|---|---|---|
| `test_cnn/cnn_window_sweep.py` | `cnn_window_sweep.csv` | Table 4, upper panel: window lengths 16, 32 and 64 under the supervised classifier |
| `Iran_new_run/window_size_check.py` | `window_sweep.log` | Table 4, lower panel: the same sweep under the one-class scorer |
| `encoding_ablation/mapping_geometry.py` | `encoding_ablation/output/mapping_geometry.csv` | Table 5: angular range, fold fraction, the derivative at both ends of the window range, and the ratio of between- to within-class spread |
| `encoding_ablation/mapping_derivation.py` | *(prints to console)* | The closed-form derivation behind Table 5, each constant printed beside its numerical check |
| `Iran_new_run/run_pr_auc/pr_auc_all_datasets_new.py` | `pr_auc_all_cells_new.csv`<br>`pr_auc_summary_by_method_new.csv`<br>`pr_auc_ens_where_best_new.csv` | Table 6: average precision for ten methods on fifteen asset-episode cells, with the random-ordering reference |
| `sensitivity_runs/bootstrap_ci.py` | `sensitivity_runs/output/bootstrap_ci.csv` | Table 7: circular moving-block bootstrap intervals for F1 and AUC on the two Iran windows |
| `sensitivity_runs/dm_overlap_corrected.py` | `sensitivity_runs/output/dm_overlap_corrected.csv` | Table 9: the pairwise Diebold-Mariano matrix with a Newey-West long-run standard error at lag 31 and the Holm adjustment across all 45 pairs |
| `backtest_new/backtest_new.py` | `backtest_new/output/backtest_new_cells.csv`<br>`backtest_new/output/bar_values_new.csv` | Table 12: the financial evaluation rebuilt on the corrected alarm-to-price alignment |
| `backtest_new/plot_summary.py` | `backtest_new/output/backtest_summary.pdf` | Figure 4: excess return under two alarm rules, beside the matched-exposure diagnostic |
| `backtest_new/plot_backtest_new.py` | `backtest_new/output/backtest_bar_*.pdf`<br>`backtest_new/output/iran_2025_2026_first_alarm_6panels_new.pdf` | Figure 3 and the per-episode bar figures |

## Analysis behind the text

| Program | Result files | What it establishes |
|---|---|---|
| `supplementary_metrics.py` | `supplementary_metrics.csv`<br>`class_counts.csv` | Supplementary Tables S1-S3: per-cell counts, class balance, and the full metric set |
| `mapping_significance.py` | *(prints to console)* | Significance between the four mappings: Newey-West DM, sign test, Friedman |
| `test_cnn/dm_test_mappings.py` | `dm_cnn_mappings.csv`<br>`dm_fanogan_mappings.csv` | Diebold-Mariano between mappings under each scorer |
| `mapping_on_fixed_pipeline.py` | `hybrid_true_vs_standalone.csv` | GAF of TadGAN scores against GAF of raw prices on the corrected pipeline |
| `Iran_new_run/test_ablation/standalone_tadgan_w32.py` | `standalone_tadgan_w32_cells.csv`<br>`standalone_tadgan_w32_summary.csv` | The TadGAN front end alone at the common window length, with a time-position control |
| `leadtime_far_matched.py` | `leadtime_far_matched.csv`<br>`paired_tests.csv` | Lead time at a matched false-alarm rate, nine operating points; none of 171 paired comparisons survives adjustment |
| `component_ablation.py` | `effective_sample_size.csv` | Effective sample size under window overlap |
| `brier_decomposition.py` | `brier_decomposition.csv` | Murphy decomposition of the Brier evaluation, which is why it no longer ranks methods |
| `sensitivity_runs/threshold_sweep.py` | `sensitivity_runs/output/threshold_sweep.csv` | Eight alarm quantiles from 0.70 to 0.99; the leading detector changes across the range |
| `sensitivity_runs/seed_variability.py` | `sensitivity_runs/output/seed_variability.csv` | Three seeds per cell; within-cell spread of 0.059 in AUC and a range up to 0.258 |
| `sensitivity_runs/overlap_check.py` | `sensitivity_runs/output/overlap_check.csv` | About four independent observations per cell under the 32-window overlap |
| `sensitivity_runs/overlap_corrected_tests.py` | `sensitivity_runs/output/overlap_corrected_tests.csv` | Block-bootstrap intervals that respect the overlap: 0.254 wide against a 0.189 spread between detectors |
| `sensitivity_runs/lstm_ablation.py` | `sensitivity_runs/output/lstm_ablation.csv`<br>`sensitivity_runs/output/lstm_structural_check.csv` | The recurrent layer receives a sequence of length one, so it carries no state between timesteps |
| `encoding_ablation/run_ablation.py` | `encoding_ablation/output/ablation_results_e5.csv`<br>`encoding_ablation/output/ablation_results_full.csv` | The six-arm encoding ablation |
| `encoding_ablation/budget_sweep.py` | `encoding_ablation/output/budget_sweep_classifier.csv` | Whether the mapping ordering depends on the training budget |
| `encoding_ablation/lambda_sweep.py` | `encoding_ablation/output/lambda_sweep_compact.csv` | Whether it depends on the gradient-penalty weight |

## Model code

| Program | What it is |
|---|---|
| `Iran_new_run/improved_tadgans_anomaly2.py` | TadGAN, the front end |
| `Iran_new_run/fanogan_four_gaf_compare.py` | The four GAF mappings and f-AnoGAN |
| `Iran_new_run/oil_baselines_gaf.py` | The eight baseline detectors |
| `Iran_new_run/ensembleExpoGAF/ensemble_expogaf.py` | The ensemble decision layer |
| `Iran_new_run/run_all_datebased.py` | The pipeline as submitted |
| `Iran_new_run/run_hybrid_true.py` | The corrected pipeline, which every reported number comes from |
| `encoding_ablation/gaf_encodings.py`<br>`encoding_ablation/models.py` | The shared loader and network definitions used by the ablation programs |

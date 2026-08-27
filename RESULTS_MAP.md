# Which program produced which table

| Program | Result files | Where it appears in the manuscript |
|---|---|---|
| `pr_auc_all_datasets_new.py` | `pr_auc_all_cells_new.csv`<br>`pr_auc_summary_by_method_new.csv`<br>`pr_auc_ens_where_best_new.csv` | PR-AUC for ten methods on fifteen asset-episode cells, with the random-ordering reference |
| `standalone_tadgan_w32.py` | `standalone_tadgan_w32_cells.csv`<br>`standalone_tadgan_w32_summary.csv` | The component ablation: TadGAN front end alone at the common window length, with a time-position control |
| `cnn_window_sweep.py` | `cnn_window_sweep.csv` | Table 4, upper panel: window lengths 16, 32 and 64 under the supervised classifier |
| `window_size_check.py` | `window_sweep.log` | Table 4, lower panel: the same sweep under the one-class scorer |
| `mapping_on_fixed_pipeline.py` | `hybrid_true_vs_standalone.csv` | GAF of TadGAN scores against GAF of raw prices on the corrected pipeline |
| `dm_test_mappings.py` | `dm_cnn_mappings.csv`<br>`dm_fanogan_mappings.csv` | Diebold-Mariano between the four angular mappings under each scorer |
| `mapping_significance.py` | *(prints to console)* | Significance between mappings: Newey-West DM, sign test, Friedman |
| `supplementary_metrics.py` | `supplementary_metrics.csv`<br>`class_counts.csv` | Supplementary Tables S1-S3: per-cell counts, class balance, and the full metric set including PR-AUC |
| `leadtime_far_matched.py` | `leadtime_far_matched.csv`<br>`paired_tests.csv` | Lead time at a matched false-alarm rate, nine operating points |
| `component_ablation.py` | `effective_sample_size.csv` | Effective sample size under window overlap |
| `brier_decomposition.py` | `brier_decomposition.csv` | Murphy decomposition of the Brier evaluation, which led to its withdrawal |

from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import optimize, stats


BASE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the ensemble decision layer within ExpoGAF-AnoNet from eight "
            "baseline anomaly-score streams and the core ExpoGAF-AnoNet stream."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE / "data" / "aligned_probabilities_9methods.csv",
        help="Aligned per-window CSV with labels and prob_EXP/prob_AT/.../prob_USAD columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "outputs",
        help="Directory for predictions, thresholds, metrics, DM matrices, and rankings.",
    )
    parser.add_argument(
        "--hard-predictions",
        type=Path,
        default=BASE / "data" / "aligned_hard_predictions_10methods.csv",
        help=(
            "Optional CSV containing the saved train-threshold pred_EXP/pred_AT/.../pred_USAD "
            "columns used for the ten-method hard-metric summary."
        ),
    )
    return parser.parse_args()


ARGS = parse_args()
INPUT = ARGS.input.resolve()
OUT = ARGS.output.resolve()
HARD_PREDICTIONS = ARGS.hard_predictions.resolve()
OUT.mkdir(parents=True, exist_ok=True)

METHODS = {
    "ExpoGAF-AnoNet (core)": "EXP",
    "Anomaly Transformer": "AT",
    "DAGMM": "DAG",
    "Deep SVDD": "DSV",
    "Isolation Forest": "IF",
    "OmniAnomaly": "OMNI",
    "One-Class SVM": "OCS",
    "TranAD": "TR",
    "USAD": "USAD",
}
ENSEMBLE = "ExpoGAF-AnoNet + ensemble layer"
ENSEMBLE_ABBR = "ENS"


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def soft_metrics(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    soft_tp = float(np.sum(p * y))
    soft_fp = float(np.sum(p * (1.0 - y)))
    soft_fn = float(np.sum((1.0 - p) * y))
    precision = soft_tp / (soft_tp + soft_fp) if soft_tp + soft_fp else 0.0
    recall = soft_tp / (soft_tp + soft_fn) if soft_tp + soft_fn else float("nan")
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def soft_threshold_f1(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    temperature: float = 0.05,
    epsilon: float = 1e-8,
) -> float:
    """Class-balanced differentiable macro-F1 for threshold selection."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    logits = np.clip((p - threshold) / temperature, -60.0, 60.0)
    soft_flag = 1.0 / (1.0 + np.exp(-logits))

    # Crash-class soft F1 (the equations reported in the manuscript).
    soft_tp = float(np.sum(soft_flag * y))
    soft_fp = float(np.sum(soft_flag * (1.0 - y)))
    soft_fn = float(np.sum((1.0 - soft_flag) * y))
    positive_f1 = 2.0 * soft_tp / (2.0 * soft_tp + soft_fp + soft_fn + epsilon)

    # Symmetric normal-class term prevents the all-crash threshold solution.
    soft_tn = float(np.sum((1.0 - soft_flag) * (1.0 - y)))
    negative_fp = float(np.sum((1.0 - soft_flag) * y))
    negative_fn = float(np.sum(soft_flag * (1.0 - y)))
    negative_f1 = 2.0 * soft_tn / (2.0 * soft_tn + negative_fp + negative_fn + epsilon)
    return 0.5 * (positive_f1 + negative_f1)


def auc_rank(y_true: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    positives = p[y == 1]
    negatives = p[y == 0]
    if not len(positives) or not len(negatives):
        return float("nan")
    ranks = stats.rankdata(np.concatenate([positives, negatives]), method="average")
    sum_positive_ranks = float(np.sum(ranks[: len(positives)]))
    u = sum_positive_ranks - len(positives) * (len(positives) + 1) / 2.0
    return u / (len(positives) * len(negatives))


def best_validation_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    temperature: float = 0.05,
) -> tuple[float, float, bool]:
    """Select the threshold by minimising differentiable soft-F1 loss."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if not np.any(y == 1):
        return 0.5, 0.0, False
    result = optimize.minimize_scalar(
        lambda threshold: 1.0 - soft_threshold_f1(y, p, float(threshold), temperature),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 2000},
    )
    threshold = float(result.x if result.success else 0.5)
    return threshold, soft_threshold_f1(y, p, threshold, temperature), bool(result.success)


def dm_test(loss_row: np.ndarray, loss_col: np.ndarray) -> tuple[float, float, float, int]:
    """Positive DM means that the row method has lower expected Brier loss."""
    difference = np.asarray(loss_col, dtype=float) - np.asarray(loss_row, dtype=float)
    difference = difference[np.isfinite(difference)]
    n = int(difference.size)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    mean_difference = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        if abs(mean_difference) <= 1e-15:
            return 0.0, 1.0, 1.0, n
        statistic = math.copysign(float("inf"), mean_difference)
        return statistic, 0.0 if statistic > 0 else 1.0, 0.0, n
    statistic = mean_difference / (sd / math.sqrt(n))
    p_one = float(1.0 - stats.t.cdf(statistic, df=n - 1))
    p_two = float(2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1)))
    return float(statistic), p_one, p_two, n


def validation_metric_weights(
    y_true: np.ndarray,
    probability_matrix: np.ndarray,
    method_keys: list[str],
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Convert historical validation metrics into shrunk soft-vote weights."""
    y = np.asarray(y_true, dtype=int)
    matrix = np.asarray(probability_matrix, dtype=float)
    reference_loss = (matrix[:, method_keys.index("EXP")] - y) ** 2
    feature_rows: list[dict[str, float | str]] = []
    for column, key in enumerate(method_keys):
        probability = matrix[:, column]
        prediction = (probability >= 0.5).astype(int)
        precision, recall, f1 = precision_recall_f1(y, prediction)
        accuracy = float(np.mean(prediction == y))
        auc = auc_rank(y, probability)
        loss = (probability - y) ** 2
        dm_statistic, _, _, _ = dm_test(loss, reference_loss)
        feature_rows.append(
            {
                "abbreviation": key,
                "validation_f1": f1,
                "validation_precision": precision,
                "validation_recall": recall,
                "validation_accuracy": accuracy,
                "validation_auc": auc,
                "validation_negative_brier": -float(np.mean(loss)),
                "validation_dm_vs_expogaf": dm_statistic,
            }
        )

    features = pd.DataFrame(feature_rows)
    feature_columns = [
        "validation_f1",
        "validation_precision",
        "validation_recall",
        "validation_accuracy",
        "validation_auc",
        "validation_negative_brier",
        "validation_dm_vs_expogaf",
    ]
    standardised = np.zeros((len(features), len(feature_columns)), dtype=float)
    for index, column in enumerate(feature_columns):
        values = features[column].to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=float(np.nanmean(values)))
        sd = float(np.std(values, ddof=0))
        standardised[:, index] = (values - float(np.mean(values))) / sd if sd > 0 else 0.0
    reliability = np.mean(standardised, axis=1)
    reliability = reliability - np.max(reliability)
    metric_weights = np.exp(reliability)
    metric_weights = metric_weights / np.sum(metric_weights)

    equal_weights = np.full(len(method_keys), 1.0 / len(method_keys), dtype=float)

    def validation_brier(shrinkage: float) -> float:
        weights = (1.0 - shrinkage) * equal_weights + shrinkage * metric_weights
        probability = matrix @ weights
        return float(np.mean((probability - y) ** 2))

    result = optimize.minimize_scalar(
        validation_brier,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 2000},
    )
    shrinkage = float(result.x if result.success else 0.0)
    trained_weights = (1.0 - shrinkage) * equal_weights + shrinkage * metric_weights
    features["standardised_reliability_score"] = reliability
    features["metric_only_weight"] = metric_weights
    features["equal_weight"] = equal_weights
    features["validation_selected_shrinkage"] = shrinkage
    features["final_vote_weight"] = trained_weights
    return trained_weights, shrinkage, features


def holm_adjust(frame: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    adjusted = frame.copy()
    adjusted["p_two_sided_holm"] = 1.0
    adjusted["significant_holm_5pct"] = False
    method_index = {name: index for index, name in enumerate(method_order)}
    for period in adjusted["period"].unique():
        period_mask = adjusted["period"] == period
        unique_mask = period_mask & adjusted.apply(
            lambda row: method_index[row["row_method"]] < method_index[row["column_method"]],
            axis=1,
        )
        unique_rows = adjusted.loc[
            unique_mask, ["row_method", "column_method", "p_two_sided"]
        ].sort_values("p_two_sided")
        running = 0.0
        holm_values: dict[tuple[str, str], float] = {}
        count = len(unique_rows)
        for rank, (_, row) in enumerate(unique_rows.iterrows()):
            candidate = min(1.0, float(row["p_two_sided"]) * (count - rank))
            running = max(running, candidate)
            pair = tuple(sorted((str(row["row_method"]), str(row["column_method"]))))
            holm_values[pair] = running
        for index in adjusted.index[period_mask]:
            row_method = str(adjusted.at[index, "row_method"])
            column_method = str(adjusted.at[index, "column_method"])
            holm = 1.0 if row_method == column_method else holm_values[
                tuple(sorted((row_method, column_method)))
            ]
            adjusted.at[index, "p_two_sided_holm"] = holm
            adjusted.at[index, "significant_holm_5pct"] = holm < 0.05
    return adjusted


aligned = pd.read_csv(INPUT)
required_columns = {"event", "asset", "window_start", "label"} | {
    f"prob_{key}" for key in METHODS.values()
} | {f"loss_{key}" for key in METHODS.values()}
missing_columns = required_columns.difference(aligned.columns)
if missing_columns:
    raise ValueError(f"{INPUT}: missing required columns {sorted(missing_columns)}")
events = list(dict.fromkeys(aligned["event"].astype(str)))
assets = list(dict.fromkeys(aligned["asset"].astype(str)))

prediction_parts: list[pd.DataFrame] = []
threshold_rows: list[dict[str, object]] = []
meta_feature_rows: list[dict[str, object]] = []
metric_rows: list[dict[str, object]] = []
key_to_method = {key: method for method, key in METHODS.items()}

for target_event in events:
    for asset in assets:
        target = aligned[(aligned["event"] == target_event) & (aligned["asset"] == asset)].copy()
        validation = aligned[(aligned["event"] != target_event) & (aligned["asset"] == asset)].copy()
        if target.empty or validation.empty:
            raise ValueError(f"Missing target or validation rows for {target_event}, {asset}")

        validation_y = validation["label"].to_numpy(dtype=int)
        method_keys = list(METHODS.values())
        validation_matrix = np.column_stack(
            [validation[f"prob_{key}"].to_numpy(dtype=float) for key in method_keys]
        )
        target_matrix = np.column_stack(
            [target[f"prob_{key}"].to_numpy(dtype=float) for key in method_keys]
        )
        vote_weights, shrinkage, meta_features = validation_metric_weights(
            validation_y, validation_matrix, method_keys
        )
        validation_probability = validation_matrix @ vote_weights
        target_probability = target_matrix @ vote_weights

        threshold, validation_soft_f1, optimiser_converged = best_validation_threshold(
            validation_y, validation_probability
        )
        target_prediction = (target_probability >= threshold).astype(int)
        target_y = target["label"].to_numpy(dtype=int)

        target["prob_ENS"] = target_probability
        target["pred_ENS"] = target_prediction
        target["loss_ENS"] = (target_probability - target_y) ** 2
        target["ensemble_threshold_from_other_events"] = threshold
        prediction_parts.append(target)

        threshold_rows.append(
            {
                "target_event": target_event,
                "asset": asset,
                "validation_events": "; ".join(event for event in events if event != target_event),
                "number_of_voters": len(method_keys),
                "validation_selected_metric_weight_shrinkage": shrinkage,
                "sigmoid_temperature": 0.05,
                "soft_f1_selected_threshold": threshold,
                "validation_soft_f1_at_threshold": validation_soft_f1,
                "threshold_optimiser_converged": optimiser_converged,
            }
        )
        for _, feature_row in meta_features.iterrows():
            row = feature_row.to_dict()
            row.update(
                {
                    "target_event": target_event,
                    "asset": asset,
                    "method": key_to_method[str(feature_row["abbreviation"])],
                    "validation_events": "; ".join(
                        event for event in events if event != target_event
                    ),
                }
            )
            meta_feature_rows.append(row)

        for method, key, probability, prediction in [
            (
                "ExpoGAF-AnoNet",
                "EXP",
                target["prob_EXP"].to_numpy(dtype=float),
                (target["prob_EXP"].to_numpy(dtype=float) >= 0.5).astype(int),
            ),
            (ENSEMBLE, ENSEMBLE_ABBR, target_probability, target_prediction),
        ]:
            precision, recall, f1 = precision_recall_f1(target_y, prediction)
            soft_precision, soft_recall, soft_f1 = soft_metrics(target_y, probability)
            metric_rows.append(
                {
                    "event": target_event,
                    "asset": asset,
                    "method": method,
                    "abbreviation": key,
                    "T": len(target),
                    "crash_windows": int(np.sum(target_y)),
                    "threshold": 0.5 if key == "EXP" else threshold,
                    "hard_precision": precision,
                    "hard_recall": recall,
                    "hard_f1": f1,
                    "probability_weighted_soft_precision": soft_precision,
                    "probability_weighted_soft_recall": soft_recall,
                    "probability_weighted_soft_f1": soft_f1,
                    "auc": auc_rank(target_y, probability),
                    "false_alarm_rate": float(np.mean(prediction[target_y == 0]))
                    if np.any(target_y == 0)
                    else float("nan"),
                    "mean_brier_loss": float(np.mean((probability - target_y) ** 2)),
                }
            )

predictions = pd.concat(prediction_parts, ignore_index=True)
predictions.to_csv(OUT / "ensemble_predictions_metric_weighted_soft_vote.csv", index=False)
thresholds = pd.DataFrame(threshold_rows)
thresholds.to_csv(OUT / "ensemble_soft_f1_thresholds_leave_one_event_out.csv", index=False)
pd.DataFrame(meta_feature_rows).to_csv(
    OUT / "ensemble_validation_metrics_and_vote_weights.csv", index=False
)
metrics = pd.DataFrame(metric_rows)
metrics.to_csv(OUT / "expogaf_ensemble_hard_and_soft_metrics.csv", index=False)

# Add the ensemble to the existing nine-method DM comparison.
all_methods = {ENSEMBLE: ENSEMBLE_ABBR, **METHODS}
periods = [*events, "Pooled all periods"]
pairwise_rows: list[dict[str, object]] = []
loss_rows: list[dict[str, object]] = []

for period in periods:
    frame = predictions if period == "Pooled all periods" else predictions[predictions["event"] == period]
    for method, key in all_methods.items():
        loss = frame[f"loss_{key}"].to_numpy(dtype=float)
        loss_rows.append(
            {
                "period": period,
                "method": method,
                "abbreviation": key,
                "mean_brier_loss": float(np.mean(loss)),
                "T": len(frame),
            }
        )
    for row_method, row_key in all_methods.items():
        row_loss = frame[f"loss_{row_key}"].to_numpy(dtype=float)
        for column_method, column_key in all_methods.items():
            column_loss = frame[f"loss_{column_key}"].to_numpy(dtype=float)
            statistic, p_one, p_two, n = dm_test(row_loss, column_loss)
            pairwise_rows.append(
                {
                    "period": period,
                    "row_method": row_method,
                    "row_abbr": row_key,
                    "column_method": column_method,
                    "column_abbr": column_key,
                    "mean_brier_row": float(np.mean(row_loss)),
                    "mean_brier_column": float(np.mean(column_loss)),
                    "loss_difference_column_minus_row": float(np.mean(column_loss - row_loss)),
                    "dm_stat_positive_row_better": statistic,
                    "p_one_sided_row_better": p_one,
                    "p_two_sided": p_two,
                    "T": n,
                }
            )

method_order = list(all_methods)
pairwise = holm_adjust(pd.DataFrame(pairwise_rows), method_order)
pairwise.to_csv(OUT / "dm_pairwise_10methods_long.csv", index=False)

loss_table = pd.DataFrame(loss_rows)
loss_table["rank_within_period"] = loss_table.groupby("period")["mean_brier_loss"].rank(
    method="min"
)
loss_table.to_csv(OUT / "mean_brier_loss_rank_10methods.csv", index=False)

abbr_order = [all_methods[method] for method in method_order]
for period in periods:
    subset = pairwise[pairwise["period"] == period]
    slug = period.lower().replace("--", "_").replace("-", "_").replace(" ", "_")
    for column, stem in [
        ("dm_stat_positive_row_better", "dm_stat_matrix"),
        ("p_one_sided_row_better", "dm_p_one_matrix"),
        ("p_two_sided_holm", "dm_holm_matrix"),
    ]:
        matrix = subset.pivot(index="row_abbr", columns="column_abbr", values=column)
        matrix = matrix.reindex(index=abbr_order, columns=abbr_order).rename_axis("row_method")
        matrix.to_csv(OUT / f"{stem}_{slug}.csv")

pooled = loss_table[loss_table["period"] == "Pooled all periods"].sort_values(
    ["mean_brier_loss", "method"]
)
print("\nPooled Brier-loss ranking (actual result):")
print(pooled[["rank_within_period", "method", "mean_brier_loss", "T"]].to_string(index=False))
print("\nCore and ensemble-layer ExpoGAF-AnoNet F1 metrics:")
print(
    metrics[
        [
            "event",
            "asset",
            "method",
            "hard_f1",
            "probability_weighted_soft_f1",
            "hard_recall",
            "auc",
            "mean_brier_loss",
        ]
    ].to_string(index=False)
)

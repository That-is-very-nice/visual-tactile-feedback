"""Tidy tables and candidate statistics for CMC migration batches."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .cmc import PAPER_CMC_METRIC
from .statistics import holm_adjust, paired_wilcoxon_signed_rank


CMC_CANDIDATE_METRICS = (
    "mean_normalized_suprathreshold_area",
    "mean_mean_suprathreshold_excess",
    "mean_band_mean",
    "max_normalized_suprathreshold_area",
    "max_mean_suprathreshold_excess",
    "max_suprathreshold_excess",
    "max_band_mean",
)


def paper_cmc_results_to_frame(
    results: Sequence[Mapping[str, object]],
    *,
    metric_name: str = PAPER_CMC_METRIC,
) -> pd.DataFrame:
    """Convert stable CMC results to one row per subject, condition, and band."""

    rows: list[dict[str, object]] = []
    for result in results:
        actual_metric = str(result.get("summary_metric", ""))
        if actual_metric != metric_name:
            raise ValueError(
                f"CMC result metric is {actual_metric!r}; expected {metric_name!r}"
            )
        values = result.get("cmc_by_band")
        if not isinstance(values, Mapping):
            raise ValueError("cmc_by_band must be a mapping")
        for band, value in values.items():
            rows.append(
                {
                    "subject": result["subject"],
                    "condition": result["condition"],
                    "event_code": result["event_code"],
                    "band": band,
                    "cmc_index": float(value),
                    "metric": metric_name,
                }
            )
    frame = pd.DataFrame(rows)
    assert_unique_cmc_rows(frame)
    return frame


def summarize_paper_cmc(
    frame: pd.DataFrame,
    *,
    visual_condition: str,
    tactile_condition: str,
    band_order: Sequence[str] = ("alpha", "beta", "gamma"),
) -> list[dict[str, object]]:
    """Summarize the frozen CMC index with paired tests and Holm correction."""

    assert_unique_cmc_rows(frame)
    required = {"subject", "condition", "band", "cmc_index"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing stable CMC columns: {missing_columns}")

    statistics: list[dict[str, object]] = []
    for band in band_order:
        subset = frame[frame["band"] == band]
        wide = subset.pivot(
            index="subject", columns="condition", values="cmc_index"
        )
        missing_conditions = [
            condition
            for condition in (visual_condition, tactile_condition)
            if condition not in wide.columns
        ]
        if missing_conditions:
            raise ValueError(f"Missing conditions for {band}: {missing_conditions}")
        paired = wide[[visual_condition, tactile_condition]].dropna()
        if len(paired) != len(wide):
            raise ValueError(f"CMC {band} contains incomplete subject pairs")
        result = paired_wilcoxon_signed_rank(
            paired[visual_condition].to_numpy(),
            paired[tactile_condition].to_numpy(),
        )
        statistics.append(
            {
                "band": band,
                "subject_count": int(len(paired)),
                "visual_mean": float(np.mean(paired[visual_condition])),
                "tactile_mean": float(np.mean(paired[tactile_condition])),
                **result.to_dict(),
            }
        )

    adjusted = holm_adjust(
        np.array([row["p_value"] for row in statistics], dtype=float)
    )
    for row, adjusted_p in zip(statistics, adjusted):
        row["p_value_holm"] = float(adjusted_p)
    return statistics


def assert_unique_cmc_rows(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str] = ("subject", "condition", "band"),
) -> None:
    """Reject duplicate analysis units before pivoting or statistical testing.

    The historical CMC statistics notebook grouped duplicate rows with
    ``mean()``. That behavior silently combined two conflicting processing
    runs and produced the values reported in Fig. 5 and Table 1. The rebuilt
    pipeline treats duplicate subject-condition-band keys as a provenance
    error instead of averaging them implicitly.
    """

    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing CMC key columns: {missing}")

    duplicate_mask = frame.duplicated(list(key_columns), keep=False)
    if not duplicate_mask.any():
        return

    counts = (
        frame.loc[duplicate_mask, list(key_columns)]
        .value_counts(sort=True)
        .rename("row_count")
        .reset_index()
    )
    preview = counts.head(3).to_dict(orient="records")
    raise ValueError(
        "CMC rows must be unique by "
        f"{list(key_columns)}; found {len(counts)} duplicated keys. "
        f"Examples: {preview}"
    )


def cmc_results_to_frame(results: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Convert nested trace results to a subject-condition-band table."""

    rows: list[dict[str, object]] = []
    for result in results:
        summaries = result["roi_candidate_metrics"]
        if not isinstance(summaries, Mapping):
            raise ValueError("roi_candidate_metrics must be a mapping")
        for band, metrics in summaries.items():
            if not isinstance(metrics, Mapping):
                raise ValueError("band metrics must be a mapping")
            rows.append(
                {
                    "subject": result["subject"],
                    "condition": result["condition"],
                    "event_code": result["event_code"],
                    "band": band,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_cmc_candidates(
    frame: pd.DataFrame,
    *,
    visual_condition: str,
    tactile_condition: str,
    band_order: Sequence[str] = ("alpha", "beta", "gamma"),
    metric_names: Sequence[str] = CMC_CANDIDATE_METRICS,
) -> list[dict[str, object]]:
    """Run paired Wilcoxon tests and Holm correction for each candidate metric."""

    assert_unique_cmc_rows(frame)

    rows: list[dict[str, object]] = []
    for metric in metric_names:
        metric_rows: list[dict[str, object]] = []
        for band in band_order:
            subset = frame[frame["band"] == band]
            wide = subset.pivot(index="subject", columns="condition", values=metric)
            missing = [
                condition
                for condition in (visual_condition, tactile_condition)
                if condition not in wide.columns
            ]
            if missing:
                raise ValueError(f"Missing conditions for {metric}/{band}: {missing}")
            result = paired_wilcoxon_signed_rank(
                wide[visual_condition].to_numpy(),
                wide[tactile_condition].to_numpy(),
            )
            metric_rows.append(
                {
                    "metric": metric,
                    "band": band,
                    "visual_mean": float(np.mean(wide[visual_condition])),
                    "tactile_mean": float(np.mean(wide[tactile_condition])),
                    **result.to_dict(),
                }
            )
        adjusted = holm_adjust(
            np.array([row["p_value"] for row in metric_rows], dtype=float)
        )
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["p_value_holm"] = float(adjusted_p)
        rows.extend(metric_rows)
    return rows

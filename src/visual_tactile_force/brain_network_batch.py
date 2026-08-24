"""Tidy brain-network tables and paired multiple-comparison procedures."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from .brain_network import PAPER_NETWORK_METRIC
from .statistics import holm_adjust, paired_wilcoxon_signed_rank


NETWORK_KEY_COLUMNS = (
    "subject",
    "condition",
    "band",
    "roi_source",
    "roi_target",
)


def assert_unique_brain_network_rows(frame: pd.DataFrame) -> None:
    """Reject repeated subject-condition-band-ROI analysis units."""

    missing = sorted(set(NETWORK_KEY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing brain-network key columns: {missing}")
    duplicate = frame.duplicated(list(NETWORK_KEY_COLUMNS), keep=False)
    if duplicate.any():
        preview = (
            frame.loc[duplicate, list(NETWORK_KEY_COLUMNS)]
            .value_counts()
            .head(3)
            .rename("row_count")
            .reset_index()
            .to_dict(orient="records")
        )
        raise ValueError(f"Brain-network rows must be unique: {preview}")


def brain_network_results_to_frame(
    results: Sequence[Mapping[str, object]],
    *,
    metric_name: str = PAPER_NETWORK_METRIC,
) -> pd.DataFrame:
    """Convert nested trace output to one row per subject/condition/ROI pair/band."""

    rows: list[dict[str, object]] = []
    for result in results:
        if str(result.get("summary_metric", "")) != metric_name:
            raise ValueError(f"Unexpected brain-network metric in {result.get('subject')}")
        connectivity = result.get("roi_connectivity")
        if not isinstance(connectivity, Sequence):
            raise ValueError("roi_connectivity must be a sequence")
        for row in connectivity:
            if not isinstance(row, Mapping):
                raise ValueError("ROI connectivity rows must be mappings")
            rows.append(
                {
                    "subject": result["subject"],
                    "condition": result["condition"],
                    "event_code": result["event_code"],
                    **row,
                    "metric": metric_name,
                }
            )
    frame = pd.DataFrame(rows)
    assert_unique_brain_network_rows(frame)
    return frame


def _paired_edge(
    frame: pd.DataFrame,
    *,
    visual_condition: str,
    tactile_condition: str,
) -> pd.DataFrame:
    wide = frame.pivot(index="subject", columns="condition", values=PAPER_NETWORK_METRIC)
    missing = [
        condition
        for condition in (visual_condition, tactile_condition)
        if condition not in wide.columns
    ]
    if missing:
        raise ValueError(f"Missing brain-network conditions: {missing}")
    paired = wide[[visual_condition, tactile_condition]].dropna()
    if len(paired) != len(wide):
        raise ValueError("Brain-network edge contains incomplete subject pairs")
    return paired


def summarize_declared_holm(
    frame: pd.DataFrame,
    *,
    visual_condition: str,
    tactile_condition: str,
) -> list[dict[str, object]]:
    """Apply paired Wilcoxon tests and one Holm family across 45 edges × 5 bands."""

    assert_unique_brain_network_rows(frame)
    interregional = frame[frame["edge_scope"] == "interregional"]
    rows: list[dict[str, object]] = []
    for (band, roi_source, roi_target), subset in interregional.groupby(
        ["band", "roi_source", "roi_target"], sort=True
    ):
        paired = _paired_edge(
            subset,
            visual_condition=visual_condition,
            tactile_condition=tactile_condition,
        )
        result = paired_wilcoxon_signed_rank(
            paired[visual_condition].to_numpy(),
            paired[tactile_condition].to_numpy(),
        )
        rows.append(
            {
                "analysis_profile": "declared_wilcoxon_holm_global_225",
                "band": band,
                "roi_source": roi_source,
                "roi_target": roi_target,
                "edge_scope": "interregional",
                "subject_count": int(len(paired)),
                "visual_mean": float(paired[visual_condition].mean()),
                "tactile_mean": float(paired[tactile_condition].mean()),
                "mean_difference": float(
                    (paired[visual_condition] - paired[tactile_condition]).mean()
                ),
                **result.to_dict(),
            }
        )
    adjusted = holm_adjust(np.array([row["p_value"] for row in rows], dtype=float))
    for row, adjusted_p in zip(rows, adjusted):
        row["p_value_holm"] = float(adjusted_p)
        row["significant"] = bool(adjusted_p < 0.05)
        row["correction_family_size"] = len(rows)
    return rows


def exact_studentized_max_t(differences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return observed paired t statistics and exhaustive sign-flip max-T p-values."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("differences must have shape (subjects >= 2, tests >= 1)")
    if not np.all(np.isfinite(values)):
        raise ValueError("differences must be finite")
    subject_count = values.shape[0]
    if subject_count > 20:
        raise ValueError("Exhaustive sign flipping is limited to 20 subjects")

    def studentized(means: np.ndarray) -> np.ndarray:
        sums_of_squares = np.sum(values**2, axis=0)
        variance = (sums_of_squares[None, :] - subject_count * means**2) / (
            subject_count - 1
        )
        variance = np.maximum(variance, 0.0)
        standard_error = np.sqrt(variance / subject_count)
        result = np.zeros_like(means)
        nonzero_error = standard_error > 0
        np.divide(means, standard_error, out=result, where=nonzero_error)
        degenerate = (~nonzero_error) & (means != 0)
        result[degenerate] = np.copysign(np.inf, means[degenerate])
        return result

    observed = studentized(values.mean(axis=0, keepdims=True))[0]
    permutation_count = 1 << subject_count
    codes = np.arange(permutation_count, dtype=np.uint64)[:, None]
    bit_positions = np.arange(subject_count, dtype=np.uint64)[None, :]
    bits = ((codes >> bit_positions) & 1).astype(np.int8)
    signs = (bits * 2 - 1).astype(float)
    permuted_means = (signs @ values) / subject_count
    permuted_t = studentized(permuted_means)
    null_max = np.max(np.abs(permuted_t), axis=1)
    p_values = np.mean(null_max[:, None] >= np.abs(observed)[None, :], axis=0)
    return observed, p_values


def summarize_published_style_max_t(
    frame: pd.DataFrame,
    *,
    visual_condition: str,
    tactile_condition: str,
) -> list[dict[str, object]]:
    """Reconstruct the actual per-band max-T family used by the historical notebook.

    The correction family contains 45 interregional and 10 within-ROI tests.
    Directional duplicates from the old CSV are omitted because they have the
    same absolute statistic and therefore cannot change the maximum.
    """

    assert_unique_brain_network_rows(frame)
    rows: list[dict[str, object]] = []
    for band, band_frame in frame.groupby("band", sort=True):
        difference_columns: list[np.ndarray] = []
        metadata: list[tuple[str, str, str, pd.DataFrame]] = []
        subjects: list[str] | None = None
        for (roi_source, roi_target, edge_scope), subset in band_frame.groupby(
            ["roi_source", "roi_target", "edge_scope"], sort=True
        ):
            paired = _paired_edge(
                subset,
                visual_condition=visual_condition,
                tactile_condition=tactile_condition,
            ).sort_index()
            current_subjects = paired.index.astype(str).tolist()
            if subjects is None:
                subjects = current_subjects
            elif current_subjects != subjects:
                raise ValueError(f"max-T subject set differs for {band}/{roi_source}/{roi_target}")
            difference_columns.append(
                paired[visual_condition].to_numpy() - paired[tactile_condition].to_numpy()
            )
            metadata.append((roi_source, roi_target, edge_scope, paired))
        matrix = np.stack(difference_columns, axis=1)
        observed, p_values = exact_studentized_max_t(matrix)
        for (roi_source, roi_target, edge_scope, paired), t_value, p_value in zip(
            metadata, observed, p_values
        ):
            differences = (
                paired[visual_condition].to_numpy()
                - paired[tactile_condition].to_numpy()
            )
            if np.allclose(differences, differences[0], rtol=0.0, atol=1e-15):
                raw_p_value = 1.0 if np.mean(differences) == 0 else 0.0
            else:
                raw_p_value = float(
                    ttest_rel(
                        paired[visual_condition].to_numpy(),
                        paired[tactile_condition].to_numpy(),
                    ).pvalue
                )
            rows.append(
                {
                    "analysis_profile": "published_style_exact_max_t_per_band_55",
                    "band": band,
                    "roi_source": roi_source,
                    "roi_target": roi_target,
                    "edge_scope": edge_scope,
                    "subject_count": len(paired),
                    "visual_mean": float(paired[visual_condition].mean()),
                    "tactile_mean": float(paired[tactile_condition].mean()),
                    "mean_difference": float(
                        (paired[visual_condition] - paired[tactile_condition]).mean()
                    ),
                    "t_statistic": float(t_value),
                    "p_value_t_raw": raw_p_value,
                    "p_value_max_t": float(p_value),
                    "significant": bool(p_value < 0.05),
                    "correction_family_size": matrix.shape[1],
                    "permutation_count": 1 << matrix.shape[0],
                }
            )
    return rows

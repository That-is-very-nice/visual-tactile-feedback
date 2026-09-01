"""Tidy brain-network tables and paired multiple-comparison procedures."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .brain_network import PAPER_NETWORK_METRIC, PAPER_ROI_CHANNELS
from .statistics import holm_adjust


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


def _brain_network_wilcoxon(differences: np.ndarray) -> tuple[float, float]:
    """Run the final paired Wilcoxon convention for one ROI connection."""

    values = np.asarray(differences, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    if values.size == 0 or np.all(values == 0):
        return 0.0, 1.0

    result = wilcoxon(
        values,
        zero_method="pratt",
        correction=True,
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def summarize_brain_network_wilcoxon_holm(
    frame: pd.DataFrame,
    *,
    visual_condition: str,
    tactile_condition: str,
) -> list[dict[str, object]]:
    """Run per-band Wilcoxon tests and Holm correction over 100 directed ROI rows."""

    assert_unique_brain_network_rows(frame)

    all_rows: list[dict[str, object]] = []

    for band, band_frame in frame.groupby("band", sort=True):
        directed_rows: list[dict[str, object]] = []
        common_subjects: list[str] | None = None

        for (roi_source, roi_target, edge_scope), subset in band_frame.groupby(
            ["roi_source", "roi_target", "edge_scope"],
            sort=True,
        ):
            paired = _paired_edge(
                subset,
                visual_condition=visual_condition,
                tactile_condition=tactile_condition,
            ).sort_index()

            current_subjects = paired.index.astype(str).tolist()
            if common_subjects is None:
                common_subjects = current_subjects
            elif current_subjects != common_subjects:
                raise ValueError(
                    f"Subject set differs for {band}/{roi_source}/{roi_target}"
                )

            differences = (
                paired[visual_condition].to_numpy()
                - paired[tactile_condition].to_numpy()
            )
            statistic, p_value_raw = _brain_network_wilcoxon(differences)

            base = {
                "analysis_profile": "wilcoxon_holm_per_band_directed_100",
                "band": str(band),
                "edge_scope": str(edge_scope),
                "subject_count": int(len(paired)),
                "visual_mean": float(paired[visual_condition].mean()),
                "tactile_mean": float(paired[tactile_condition].mean()),
                "mean_difference": float(np.mean(differences)),
                "wilcoxon_statistic": statistic,
                "p_value_raw": p_value_raw,
            }

            directed_rows.append(
                {
                    **base,
                    "roi_source": str(roi_source),
                    "roi_target": str(roi_target),
                }
            )

            if edge_scope == "interregional":
                directed_rows.append(
                    {
                        **base,
                        "roi_source": str(roi_target),
                        "roi_target": str(roi_source),
                    }
                )

        if len(directed_rows) != 100:
            raise ValueError(
                f"Band {band!r} produced {len(directed_rows)} directed ROI rows; expected 100"
            )

        adjusted = holm_adjust(
            np.asarray([row["p_value_raw"] for row in directed_rows], dtype=float)
        )

        for row, adjusted_p in zip(directed_rows, adjusted):
            row["p_value_holm"] = float(adjusted_p)
            row["significant"] = bool(adjusted_p < 0.05)
            row["correction_family_size"] = 100

        all_rows.extend(directed_rows)

    return sorted(
        all_rows,
        key=lambda row: (
            str(row["band"]),
            float(row["p_value_holm"]),
            str(row["roi_source"]),
            str(row["roi_target"]),
        ),
    )


def collapse_directed_brain_network_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    tolerance: float = 1e-12,
) -> list[dict[str, object]]:
    """Collapse symmetric ROI rows only after the 100-test Holm correction."""

    roi_order = {roi: index for index, roi in enumerate(PAPER_ROI_CHANNELS)}
    frame = pd.DataFrame(rows)

    canonical_source = []
    canonical_target = []

    for source, target in zip(frame["roi_source"], frame["roi_target"]):
        source = str(source)
        target = str(target)
        if roi_order[source] <= roi_order[target]:
            canonical_source.append(source)
            canonical_target.append(target)
        else:
            canonical_source.append(target)
            canonical_target.append(source)

    frame = frame.assign(
        canonical_source=canonical_source,
        canonical_target=canonical_target,
    )

    collapsed: list[dict[str, object]] = []

    for (band, source, target), group in frame.groupby(
        ["band", "canonical_source", "canonical_target"],
        sort=True,
    ):
        for column in ("mean_difference", "p_value_raw", "p_value_holm"):
            if column not in group.columns or group[column].isna().all():
                continue
            if float(group[column].max() - group[column].min()) > tolerance:
                raise ValueError(
                    f"Symmetric rows disagree for {band}/{source}/{target}/{column}"
                )

        first = group.iloc[0]
        collapsed.append(
            {
                "analysis_profile": "wilcoxon_holm_per_band_directed_100",
                "band": str(band),
                "roi_source": str(source),
                "roi_target": str(target),
                "edge_scope": (
                    "within_roi" if source == target else "interregional"
                ),
                "subject_count": int(first["subject_count"]),
                "visual_mean": float(first.get("visual_mean", np.nan)),
                "tactile_mean": float(first.get("tactile_mean", np.nan)),
                "mean_difference": float(group["mean_difference"].mean()),
                "wilcoxon_statistic": float(first.get("wilcoxon_statistic", np.nan)),
                "p_value_raw": float(first.get("p_value_raw", np.nan)),
                "p_value_holm": float(group["p_value_holm"].mean()),
                "significant": bool(float(group["p_value_holm"].mean()) < 0.05),
                "correction_family_size": 100,
                "directed_rows_collapsed": int(len(group)),
            }
        )

    return collapsed

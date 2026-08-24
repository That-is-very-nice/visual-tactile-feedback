"""Tidy tables, paired tests, and behavior correlations for PDC."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .pdc import PAPER_PDC_METRIC, PDC_DIRECTIONS
from .statistics import holm_adjust, paired_wilcoxon_signed_rank


def assert_unique_pdc_rows(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str] = ("subject", "condition", "direction", "band"),
) -> None:
    """Reject repeated analysis units instead of silently aggregating them."""

    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing PDC key columns: {missing}")
    duplicate = frame.duplicated(list(key_columns), keep=False)
    if duplicate.any():
        preview = (
            frame.loc[duplicate, list(key_columns)]
            .value_counts()
            .head(3)
            .rename("row_count")
            .reset_index()
            .to_dict(orient="records")
        )
        raise ValueError(f"PDC rows must be unique by {list(key_columns)}: {preview}")


def paper_pdc_results_to_frame(
    results: Sequence[Mapping[str, object]],
    *,
    metric_name: str = PAPER_PDC_METRIC,
) -> pd.DataFrame:
    """Convert trace results to one row per subject/condition/direction/band."""

    rows: list[dict[str, object]] = []
    for result in results:
        actual_metric = str(result.get("summary_metric", ""))
        if actual_metric != metric_name:
            raise ValueError(f"PDC result metric is {actual_metric!r}; expected {metric_name!r}")
        directions = result.get("pdc_by_direction_band")
        if not isinstance(directions, Mapping):
            raise ValueError("pdc_by_direction_band must be a mapping")
        for direction, band_values in directions.items():
            if direction not in PDC_DIRECTIONS or not isinstance(band_values, Mapping):
                raise ValueError(f"Invalid PDC direction payload: {direction!r}")
            for band, value in band_values.items():
                rows.append(
                    {
                        "subject": result["subject"],
                        "condition": result["condition"],
                        "event_code": result["event_code"],
                        "direction": direction,
                        "band": band,
                        "pdc_index": float(value),
                        "metric": metric_name,
                    }
                )
    frame = pd.DataFrame(rows)
    assert_unique_pdc_rows(frame)
    return frame


def summarize_paper_pdc(
    frame: pd.DataFrame,
    *,
    visual_condition: str,
    tactile_condition: str,
    directions: Sequence[str] = ("descending", "ascending"),
    band_order: Sequence[str] = ("alpha", "beta", "gamma"),
) -> list[dict[str, object]]:
    """Run paired Wilcoxon tests with Holm correction within each direction."""

    assert_unique_pdc_rows(frame)
    rows: list[dict[str, object]] = []
    for direction in directions:
        direction_rows: list[dict[str, object]] = []
        for band in band_order:
            subset = frame[(frame["direction"] == direction) & (frame["band"] == band)]
            wide = subset.pivot(index="subject", columns="condition", values="pdc_index")
            missing = [
                condition
                for condition in (visual_condition, tactile_condition)
                if condition not in wide.columns
            ]
            if missing:
                raise ValueError(f"Missing conditions for {direction}/{band}: {missing}")
            paired = wide[[visual_condition, tactile_condition]].dropna()
            if len(paired) != len(wide):
                raise ValueError(f"PDC {direction}/{band} contains incomplete pairs")
            result = paired_wilcoxon_signed_rank(
                paired[visual_condition].to_numpy(), paired[tactile_condition].to_numpy()
            )
            direction_rows.append(
                {
                    "direction": direction,
                    "band": band,
                    "subject_count": int(len(paired)),
                    "visual_mean": float(np.mean(paired[visual_condition])),
                    "tactile_mean": float(np.mean(paired[tactile_condition])),
                    **result.to_dict(),
                }
            )
        adjusted = holm_adjust(np.array([row["p_value"] for row in direction_rows]))
        for row, adjusted_p in zip(direction_rows, adjusted):
            row["p_value_holm"] = float(adjusted_p)
        rows.extend(direction_rows)
    return rows


def correlate_pdc_with_behavior(
    pdc: pd.DataFrame,
    behavior: pd.DataFrame,
    *,
    direction: str = "descending",
    band: str = "gamma",
    behavior_column: str = "force_cv",
) -> list[dict[str, object]]:
    """Calculate within-condition Spearman correlations used by Figure 8."""

    assert_unique_pdc_rows(pdc)
    required = {"subject", "condition", behavior_column}
    missing = sorted(required - set(behavior.columns))
    if missing:
        raise ValueError(f"Missing behavior columns: {missing}")
    if behavior.duplicated(["subject", "condition"]).any():
        raise ValueError("Behavior rows must be unique by subject and condition")
    selected = pdc[(pdc["direction"] == direction) & (pdc["band"] == band)]
    merged = selected.merge(
        behavior[["subject", "condition", behavior_column]],
        on=["subject", "condition"],
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for condition, subset in merged.groupby("condition", sort=True):
        if len(subset) < 3:
            raise ValueError(f"At least three pairs are required for {condition}")
        result = spearmanr(subset["pdc_index"], subset[behavior_column])
        rows.append(
            {
                "condition": condition,
                "direction": direction,
                "band": band,
                "behavior_metric": behavior_column,
                "n": int(len(subset)),
                "spearman_rho": float(result.statistic),
                "spearman_p": float(result.pvalue),
            }
        )
    return rows

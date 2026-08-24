"""Explicit reproduction of the historical published CMC aggregation.

This module is intentionally separate from the corrected paper-method
pipeline. It documents a legacy provenance failure: two conflicting CMC runs
were appended to one CSV, then averaged implicitly by a plotting/statistics
notebook. The resulting ``cmc5`` values reproduce published Fig. 5 and Table 1.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .statistics import holm_adjust, paired_wilcoxon_signed_rank


PUBLISHED_LEGACY_METRIC = "cmc5"
PUBLISHED_LEGACY_METRIC_DEFINITION = "ROI maximum of mean suprathreshold excess"


def reproduce_published_cmc_aggregate(
    frame: pd.DataFrame,
    *,
    subjects: Sequence[str] | None = None,
    condition_column: str = "event",
    visual_condition: str = "st_no",
    tactile_condition: str = "st_tf2",
    bands: Sequence[str] = ("alpha", "beta", "gamma"),
    metric: str = PUBLISHED_LEGACY_METRIC,
    expected_rows_per_key: int = 2,
) -> dict[str, object]:
    """Reproduce the published legacy aggregate without hiding its provenance.

    This function is a historical regression adapter, not a recommended
    analysis. It requires the duplicated input structure observed in the
    archived CSV, verifies that the duplicate values conflict, and only then
    performs the exact group-mean operation used by the legacy notebook.
    """

    key_columns = ["subject", condition_column, "band"]
    required = {*key_columns, metric}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing legacy CMC columns: {missing}")

    selected = frame[
        frame[condition_column].isin([visual_condition, tactile_condition])
        & frame["band"].isin(bands)
    ].copy()
    if subjects is not None:
        selected = selected[selected["subject"].isin(subjects)]
    if selected.empty:
        raise ValueError("No rows match the requested legacy CMC analysis")

    counts = selected.groupby(key_columns, sort=True).size()
    unexpected_counts = counts[counts != expected_rows_per_key]
    if not unexpected_counts.empty:
        preview = unexpected_counts.head(3).to_dict()
        raise ValueError(
            f"Expected {expected_rows_per_key} rows per legacy CMC key; "
            f"found {len(unexpected_counts)} keys with other counts: {preview}"
        )

    value_counts = selected.groupby(key_columns, sort=True)[metric].nunique(
        dropna=False
    )
    conflicting_key_count = int((value_counts > 1).sum())
    if conflicting_key_count != len(value_counts):
        raise ValueError(
            "Historical regression requires every duplicated key to contain "
            "conflicting values; refusing to treat identical copies as runs"
        )

    # This line deliberately mirrors the legacy pivot helper's implicit mean.
    grouped = selected.groupby(key_columns, as_index=False, sort=True)[metric].mean()

    statistics: list[dict[str, object]] = []
    for band in bands:
        subset = grouped[grouped["band"] == band]
        wide = subset.pivot(
            index="subject", columns=condition_column, values=metric
        )
        missing_conditions = [
            condition
            for condition in (visual_condition, tactile_condition)
            if condition not in wide.columns
        ]
        if missing_conditions:
            raise ValueError(f"Missing conditions for {band}: {missing_conditions}")
        if subjects is not None and len(wide) != len(subjects):
            raise ValueError(
                f"Expected {len(subjects)} subjects for {band}; found {len(wide)}"
            )
        result = paired_wilcoxon_signed_rank(
            wide[visual_condition].to_numpy(),
            wide[tactile_condition].to_numpy(),
        )
        statistics.append(
            {
                "band": band,
                "visual_mean": float(np.mean(wide[visual_condition])),
                "tactile_mean": float(np.mean(wide[tactile_condition])),
                **result.to_dict(),
            }
        )

    adjusted = holm_adjust(
        np.array([row["p_value"] for row in statistics], dtype=float)
    )
    for row, adjusted_p in zip(statistics, adjusted):
        row["p_value_holm"] = float(adjusted_p)

    return {
        "status": "historical_published_output_reproduced",
        "warning": (
            "This reproduces a published legacy artifact by explicitly averaging "
            "two conflicting appended processing runs. It is not the corrected "
            "paper-method CMC pipeline."
        ),
        "metric": metric,
        "metric_definition": PUBLISHED_LEGACY_METRIC_DEFINITION,
        "difference_definition": f"{visual_condition} minus {tactile_condition}",
        "aggregation": "arithmetic mean across conflicting duplicate rows",
        "source_row_count": int(len(selected)),
        "unique_key_count": int(len(counts)),
        "rows_per_key": expected_rows_per_key,
        "conflicting_key_count": conflicting_key_count,
        "statistics": statistics,
    }

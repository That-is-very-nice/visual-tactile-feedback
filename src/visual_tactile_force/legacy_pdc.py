"""Auditable adapters for the PDC values actually used in Table 2 and Figure 8."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

from .statistics import holm_adjust, paired_wilcoxon_signed_rank


PUBLISHED_LEGACY_COLUMNS = {
    "descending": "pdc_down5",
    "ascending": "pdc_up5",
}
PUBLISHED_LEGACY_METRIC_DEFINITION = (
    "ROI maximum of channel-wise mean positive PDC excess above a scalar null threshold"
)


def reproduce_published_pdc_statistics(
    frame: pd.DataFrame,
    *,
    subjects: Sequence[str] | None = None,
    condition_column: str = "event",
    visual_condition: str = "st_no",
    tactile_condition: str = "st_tf2",
    bands: Sequence[str] = ("alpha", "beta", "gamma"),
) -> dict[str, object]:
    """Reproduce Table 2 from the unique historical PDC5 rows."""

    required = {"subject", condition_column, "band", *PUBLISHED_LEGACY_COLUMNS.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing legacy PDC columns: {missing}")
    selected = frame[
        frame[condition_column].isin([visual_condition, tactile_condition])
        & frame["band"].isin(bands)
    ].copy()
    if subjects is not None:
        selected = selected[selected["subject"].isin(subjects)]
    keys = ["subject", condition_column, "band"]
    if selected.duplicated(keys).any():
        raise ValueError("Historical PDC rows must be unique; duplicate keys were found")

    statistics: list[dict[str, object]] = []
    for direction, metric in PUBLISHED_LEGACY_COLUMNS.items():
        direction_rows: list[dict[str, object]] = []
        for band in bands:
            subset = selected[selected["band"] == band]
            wide = subset.pivot(index="subject", columns=condition_column, values=metric)
            if subjects is not None and len(wide) != len(subjects):
                raise ValueError(f"Expected {len(subjects)} subjects for {band}; found {len(wide)}")
            differences = wide[visual_condition].to_numpy() - wide[tactile_condition].to_numpy()
            stable = paired_wilcoxon_signed_rank(
                wide[visual_condition].to_numpy(), wide[tactile_condition].to_numpy()
            )
            # The paper-era SciPy behavior switched to a normal approximation
            # when zero differences existed. Calling with zeros retained is
            # required to reproduce the published ascending-alpha p-value.
            legacy_test = wilcoxon(
                differences,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="auto",
            )
            direction_rows.append(
                {
                    "direction": direction,
                    "band": band,
                    "subject_count": int(len(wide)),
                    "visual_mean": float(np.mean(wide[visual_condition])),
                    "tactile_mean": float(np.mean(wide[tactile_condition])),
                    **stable.to_dict(),
                    "statistic": float(legacy_test.statistic),
                    "p_value": float(legacy_test.pvalue),
                }
            )
        adjusted = holm_adjust(np.array([row["p_value"] for row in direction_rows]))
        for row, adjusted_p in zip(direction_rows, adjusted):
            row["p_value_holm"] = float(adjusted_p)
        statistics.extend(direction_rows)

    return {
        "status": "historical_published_output_reproduced",
        "warning": (
            "This adapter reproduces the PDC5 values used in Table 2. PDC5 differs "
            "from the normalized-area metric described in the manuscript method."
        ),
        "metrics": dict(PUBLISHED_LEGACY_COLUMNS),
        "metric_definition": PUBLISHED_LEGACY_METRIC_DEFINITION,
        "difference_definition": f"{visual_condition} minus {tactile_condition}",
        "source_row_count": int(len(selected)),
        "statistics": statistics,
    }


def reproduce_published_pdc_behavior_correlations(
    frame: pd.DataFrame,
    *,
    band: str = "gamma",
    conditions: Sequence[str] = ("st_no", "st_tf2"),
) -> dict[str, object]:
    """Recompute the Figure 8 correlations from its archived merged-pair table."""

    required = {"subject", "event", "band", "pdc_down5", "fluct_across_trials"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing legacy correlation columns: {missing}")
    selected = frame[(frame["band"] == band) & frame["event"].isin(conditions)].copy()
    if selected.duplicated(["subject", "event", "band"]).any():
        raise ValueError("Legacy correlation pairs must be unique")
    correlations: list[dict[str, object]] = []
    for condition in conditions:
        subset = selected[selected["event"] == condition]
        pearson = pearsonr(subset["pdc_down5"], subset["fluct_across_trials"])
        spearman = spearmanr(subset["pdc_down5"], subset["fluct_across_trials"])
        correlations.append(
            {
                "condition": condition,
                "band": band,
                "n": int(len(subset)),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        )
    return {
        "status": "historical_published_output_reproduced",
        "warning": (
            "Figure 8 used archived PDC5 plus a historical force-variability table; "
            "the latter is not byte-identical to the rebuilt behavior output."
        ),
        "correlations": correlations,
    }

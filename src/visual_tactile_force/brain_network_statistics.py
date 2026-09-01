"""Load and normalize saved brain-network statistical results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .brain_network import PAPER_BANDS_HZ, PAPER_ROI_CHANNELS


def load_saved_brain_network_statistics(path: str | Path) -> pd.DataFrame:
    """Load the complete saved Wilcoxon-Holm output into the stable schema."""

    source = pd.read_csv(path)
    required = {"frequency", "source", "target", "mean_diff", "p_maxT"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Brain-network statistical table is missing columns: {missing}")
    if "event_group" in source.columns:
        source = source[source["event_group"] == "fb"].copy()
    if "comparison" in source.columns:
        source = source[source["comparison"] == "st_no vs st_tf2"].copy()

    rows: list[dict[str, object]] = []
    for row in source.to_dict(orient="records"):
        roi_source = str(row["source"])
        roi_target = str(row["target"])
        adjusted_p = float(row["p_maxT"])
        rows.append(
            {
                "analysis_profile": "wilcoxon_holm_per_band_directed_100",
                "band": str(row["frequency"]),
                "roi_source": roi_source,
                "roi_target": roi_target,
                "edge_scope": "within_roi" if roi_source == roi_target else "interregional",
                "subject_count": 15,
                "visual_mean": float("nan"),
                "tactile_mean": float("nan"),
                "mean_difference": float(row["mean_diff"]),
                "wilcoxon_statistic": float("nan"),
                "p_value_raw": float("nan"),
                "p_value_holm": adjusted_p,
                "significant": bool(adjusted_p < 0.05),
                "correction_family_size": 100,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 500:
        raise ValueError(f"Saved brain-network statistics contain {len(frame)} rows; expected 500")
    counts = frame.groupby("band").size().to_dict()
    expected_counts = {band: 100 for band in PAPER_BANDS_HZ}
    if counts != expected_counts:
        raise ValueError(f"Expected 100 saved statistical rows per band; found {counts}")
    if frame.duplicated(["band", "roi_source", "roi_target"]).any():
        raise ValueError("Saved brain-network statistics contain duplicate directed ROI rows")
    expected_pairs = {
        (source_roi, target_roi)
        for source_roi in PAPER_ROI_CHANNELS
        for target_roi in PAPER_ROI_CHANNELS
    }
    for band, group in frame.groupby("band"):
        actual_pairs = set(zip(group["roi_source"], group["roi_target"]))
        if actual_pairs != expected_pairs:
            raise ValueError(f"Band {band!r} does not contain the complete 10 x 10 ROI grid")
    if not np.all(np.isfinite(frame["mean_difference"])):
        raise ValueError("Saved brain-network mean differences must be finite")
    if not frame["p_value_holm"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("Saved Holm-adjusted p-values must lie between zero and one")
    return frame

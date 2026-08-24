"""Adapters for the historical brain-network CSV outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .brain_network import PAPER_NETWORK_METRIC, PAPER_ROI_CHANNELS
from .brain_network_batch import assert_unique_brain_network_rows


def _canonical_roi_pair(source: str, target: str) -> tuple[str, str]:
    order = {roi: index for index, roi in enumerate(PAPER_ROI_CHANNELS)}
    if source not in order or target not in order:
        raise ValueError(f"Unknown historical ROI pair: {source!r}, {target!r}")
    return (source, target) if order[source] <= order[target] else (target, source)


def load_legacy_roi_connectivity(
    path: str | Path,
    *,
    conditions: Sequence[str] = ("st_no", "st_tf2"),
) -> pd.DataFrame:
    """Normalize the 20,250-row historical ROI table to the stable schema."""

    source = pd.read_csv(path)
    required = {"subject", "event", "band", "sreg", "treg", "n_edges", "coherence"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Historical ROI table is missing columns: {missing}")
    source = source[source["event"].isin(conditions)].copy()
    pairs = [
        _canonical_roi_pair(str(left), str(right))
        for left, right in zip(source["sreg"], source["treg"])
    ]
    frame = pd.DataFrame(
        {
            "subject": source["subject"].astype(str),
            "condition": source["event"].astype(str),
            "event_code": np.nan,
            "band": source["band"].astype(str),
            "roi_source": [pair[0] for pair in pairs],
            "roi_target": [pair[1] for pair in pairs],
            "edge_scope": "interregional",
            "channel_pair_count": (source["n_edges"].astype(int) // 2),
            PAPER_NETWORK_METRIC: source["coherence"].abs().astype(float),
            "metric": PAPER_NETWORK_METRIC,
        }
    )
    assert_unique_brain_network_rows(frame)
    return frame


def load_published_max_t(path: str | Path) -> pd.DataFrame:
    """Collapse directed duplicates in the exact max-T table used by the paper."""

    source = pd.read_csv(path)
    required = {"frequency", "source", "target", "mean_diff", "p_maxT"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Historical max-T table is missing columns: {missing}")
    if "event_group" in source:
        source = source[source["event_group"] == "fb"]
    if "comparison" in source:
        source = source[source["comparison"] == "st_no vs st_tf2"]
    pairs = [
        _canonical_roi_pair(str(left), str(right))
        for left, right in zip(source["source"], source["target"])
    ]
    source = source.assign(
        roi_source=[pair[0] for pair in pairs],
        roi_target=[pair[1] for pair in pairs],
    )
    rows: list[dict[str, object]] = []
    for (band, roi_source, roi_target), group in source.groupby(
        ["frequency", "roi_source", "roi_target"], sort=True
    ):
        for column in ("mean_diff", "p_maxT"):
            if float(group[column].max() - group[column].min()) > 1e-12:
                raise ValueError(
                    "Directed historical duplicates disagree for "
                    f"{band}/{roi_source}/{roi_target}"
                )
        rows.append(
            {
                "analysis_profile": "published_legacy_exact_max_t",
                "band": str(band),
                "roi_source": roi_source,
                "roi_target": roi_target,
                "edge_scope": "within_roi" if roi_source == roi_target else "interregional",
                "mean_difference": float(group["mean_diff"].mean()),
                "p_value_max_t": float(group["p_maxT"].mean()),
                "significant": bool(float(group["p_maxT"].mean()) < 0.05),
                "directed_rows_collapsed": int(len(group)),
            }
        )
    return pd.DataFrame(rows)

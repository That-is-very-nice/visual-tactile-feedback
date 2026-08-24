"""Compact paper-facing figures for ROI absolute imaginary coherence."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .brain_network import PAPER_NETWORK_METRIC, PAPER_ROI_CHANNELS


def _matrix(
    frame: pd.DataFrame,
    *,
    value_column: str,
    roi_order: Sequence[str],
) -> np.ndarray:
    values = np.full((len(roi_order), len(roi_order)), np.nan, dtype=float)
    index = {roi: position for position, roi in enumerate(roi_order)}
    for row in frame.to_dict(orient="records"):
        source = index[str(row["roi_source"])]
        target = index[str(row["roi_target"])]
        value = float(row[value_column])
        values[source, target] = value
        values[target, source] = value
    return values


def _save(fig: plt.Figure, output_paths: Sequence[str | Path]) -> None:
    for path_value in output_paths:
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_condition_network_matrices(
    summary: pd.DataFrame,
    *,
    visual_condition: str,
    tactile_condition: str,
    band_order: Sequence[str],
    output_paths: Sequence[str | Path],
) -> None:
    """Plot mean ROI matrices for visual and tactile feedback (paper Figure 9 role)."""

    interregional = summary[summary["edge_scope"] == "interregional"]
    means = (
        interregional.groupby(["condition", "band", "roi_source", "roi_target"], as_index=False)[
            PAPER_NETWORK_METRIC
        ].mean()
    )
    roi_order = list(PAPER_ROI_CHANNELS)
    matrices = []
    for condition in (visual_condition, tactile_condition):
        for band in band_order:
            matrices.append(
                _matrix(
                    means[(means["condition"] == condition) & (means["band"] == band)],
                    value_column=PAPER_NETWORK_METRIC,
                    roi_order=roi_order,
                )
            )
    vmax = float(np.nanpercentile(np.concatenate([matrix.ravel() for matrix in matrices]), 99))
    fig, axes = plt.subplots(2, len(band_order), figsize=(3.2 * len(band_order), 6.6))
    last = None
    labels = [roi.replace("Left_", "L-").replace("Right_", "R-") for roi in roi_order]
    for row_index, condition in enumerate((visual_condition, tactile_condition)):
        for column_index, band in enumerate(band_order):
            ax = axes[row_index, column_index]
            matrix = matrices[row_index * len(band_order) + column_index]
            last = ax.imshow(matrix, vmin=0.0, vmax=vmax, cmap="viridis")
            ax.set_title(band.capitalize())
            if column_index == 0:
                ax.set_ylabel("Visual" if condition == visual_condition else "Tactile")
                ax.set_yticks(range(len(labels)), labels=labels, fontsize=6)
            else:
                ax.set_yticks([])
            if row_index == 1:
                ax.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=6)
            else:
                ax.set_xticks([])
    if last is not None:
        fig.colorbar(last, ax=axes, fraction=0.018, pad=0.02, label="Absolute ImCoh")
    fig.suptitle("ROI functional connectivity")
    _save(fig, output_paths)


def plot_difference_matrices(
    statistics: pd.DataFrame,
    *,
    band_order: Sequence[str],
    p_column: str,
    output_paths: Sequence[str | Path],
) -> None:
    """Plot visual-minus-tactile matrices and outline corrected significant edges."""

    interregional = statistics[statistics["edge_scope"] == "interregional"]
    roi_order = list(PAPER_ROI_CHANNELS)
    matrices = [
        _matrix(
            interregional[interregional["band"] == band],
            value_column="mean_difference",
            roi_order=roi_order,
        )
        for band in band_order
    ]
    limit = float(np.nanmax(np.abs(np.concatenate([matrix.ravel() for matrix in matrices]))))
    fig, axes = plt.subplots(1, len(band_order), figsize=(3.2 * len(band_order), 3.8))
    labels = [roi.replace("Left_", "L-").replace("Right_", "R-") for roi in roi_order]
    last = None
    for index, (band, matrix) in enumerate(zip(band_order, matrices)):
        ax = axes[index]
        last = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap="RdBu_r")
        ax.set_title(band.capitalize())
        subset = interregional[interregional["band"] == band]
        roi_index = {roi: position for position, roi in enumerate(roi_order)}
        for row in subset[subset[p_column] < 0.05].to_dict(orient="records"):
            left = roi_index[str(row["roi_source"])]
            right = roi_index[str(row["roi_target"])]
            for y, x in ((left, right), (right, left)):
                ax.add_patch(
                    plt.Rectangle(
                        (x - 0.46, y - 0.46),
                        0.92,
                        0.92,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.1,
                    )
                )
        if index == 0:
            ax.set_yticks(range(len(labels)), labels=labels, fontsize=6)
        else:
            ax.set_yticks([])
        ax.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=6)
    if last is not None:
        fig.colorbar(last, ax=axes, fraction=0.018, pad=0.02, label="Visual − tactile")
    correction_label = "exact max-T" if p_column == "p_value_max_t" else "Holm"
    fig.suptitle(
        f"ROI connectivity differences; black outline = {correction_label} p < .05"
    )
    _save(fig, output_paths)

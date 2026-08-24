"""Figures for the corrected CMC method output."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def plot_cmc_summary(
    summary_table: pd.DataFrame,
    statistics: Sequence[Mapping[str, object]],
    *,
    visual_condition: str,
    tactile_condition: str,
    output_paths: Sequence[str | Path],
    band_order: Sequence[str] = ("alpha", "beta", "gamma"),
) -> None:
    """Plot paired subject values and group means for the stable CMC index."""

    import matplotlib.pyplot as plt

    required = {"subject", "condition", "band", "cmc_index"}
    missing = sorted(required - set(summary_table.columns))
    if missing:
        raise ValueError(f"Missing CMC figure columns: {missing}")
    statistics_by_band = {str(row["band"]): row for row in statistics}

    colors = ("#4C78A8", "#E45756")
    fig, axes = plt.subplots(1, len(band_order), figsize=(9.0, 3.4), sharey=True)
    axes_values = np.atleast_1d(axes)
    for axis, band in zip(axes_values, band_order):
        subset = summary_table[summary_table["band"] == band]
        wide = subset.pivot(
            index="subject", columns="condition", values="cmc_index"
        )
        if not {visual_condition, tactile_condition}.issubset(wide.columns):
            raise ValueError(f"Missing conditions for CMC figure band {band}")
        paired = wide[[visual_condition, tactile_condition]].dropna()
        values = paired.to_numpy(dtype=float)
        for row in values:
            axis.plot((0, 1), row, color="#9E9E9E", alpha=0.45, linewidth=0.8)
        axis.scatter(
            np.zeros(len(values)), values[:, 0], color=colors[0], s=24, zorder=3
        )
        axis.scatter(
            np.ones(len(values)), values[:, 1], color=colors[1], s=24, zorder=3
        )
        means = np.mean(values, axis=0)
        sem = np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
        axis.errorbar(
            (0, 1),
            means,
            yerr=sem,
            fmt="D",
            color="#222222",
            markersize=5,
            capsize=3,
            linewidth=1.2,
            zorder=4,
        )
        adjusted_p = float(statistics_by_band[band]["p_value_holm"])
        axis.set_title(f"{band.capitalize()}\nHolm p={adjusted_p:.3g}")
        axis.set_xticks((0, 1), ("Visual", "Tactile"))
        axis.set_xlim(-0.35, 1.35)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        for spine_name in ("top", "right"):
            axis.spines[spine_name].set_visible(False)

    axes_values[0].set_ylabel("CMC index")
    fig.suptitle("Corrected single-run CMC method", fontsize=12)
    fig.tight_layout()
    for output_path_value in output_paths:
        output_path = Path(output_path_value)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

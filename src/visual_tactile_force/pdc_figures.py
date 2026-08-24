"""Deterministic figures for corrected PDC condition effects and correlations."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pdc_summary(
    frame: pd.DataFrame,
    statistics: Sequence[Mapping[str, object]],
    *,
    visual_condition: str,
    tactile_condition: str,
    output_paths: Sequence[Path],
) -> None:
    """Plot paired subject values for three bands in both directions."""

    bands = ("alpha", "beta", "gamma")
    directions = ("descending", "ascending")
    statistic_index = {
        (str(row["direction"]), str(row["band"])): row for row in statistics
    }
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.5), sharey=False)
    for row_index, direction in enumerate(directions):
        for column_index, band in enumerate(bands):
            axis = axes[row_index, column_index]
            subset = frame[(frame["direction"] == direction) & (frame["band"] == band)]
            wide = subset.pivot(index="subject", columns="condition", values="pdc_index")
            x = np.array([0.0, 1.0])
            for _, subject in wide.iterrows():
                axis.plot(
                    x,
                    [subject[visual_condition], subject[tactile_condition]],
                    color="#8c8c8c",
                    alpha=0.55,
                    linewidth=0.8,
                )
            means = [wide[visual_condition].mean(), wide[tactile_condition].mean()]
            axis.plot(x, means, color="#222222", marker="o", linewidth=2.2)
            axis.set_xticks(x, ["Visual", "Tactile"])
            axis.set_title(band.capitalize())
            if column_index == 0:
                axis.set_ylabel(f"{direction.capitalize()} PDC index")
            statistic = statistic_index[(direction, band)]
            axis.text(
                0.5,
                0.98,
                f"Holm p={float(statistic['p_value_holm']):.3g}",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )
            axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_pdc_behavior_correlation(
    pdc: pd.DataFrame,
    behavior: pd.DataFrame,
    correlations: Sequence[Mapping[str, object]],
    *,
    behavior_column: str,
    output_paths: Sequence[Path],
) -> None:
    """Plot corrected descending-gamma PDC against behavior by condition."""

    selected = pdc[(pdc["direction"] == "descending") & (pdc["band"] == "gamma")]
    merged = selected.merge(
        behavior[["subject", "condition", behavior_column]],
        on=["subject", "condition"],
        how="inner",
        validate="one_to_one",
    )
    conditions = sorted(merged["condition"].unique())
    correlation_index = {str(row["condition"]): row for row in correlations}
    condition_labels = {"st_no": "Visual", "st_tf2": "Tactile"}
    figure, axes = plt.subplots(1, len(conditions), figsize=(7.2, 3.3), squeeze=False)
    for axis, condition in zip(axes[0], conditions):
        subset = merged[merged["condition"] == condition]
        axis.scatter(subset[behavior_column], subset["pdc_index"], color="#3a6ea5", s=28)
        if len(subset) >= 2:
            coefficient = np.polyfit(subset[behavior_column], subset["pdc_index"], 1)
            x = np.linspace(subset[behavior_column].min(), subset[behavior_column].max(), 100)
            axis.plot(x, np.polyval(coefficient, x), color="#3a6ea5", linewidth=1.2)
        statistic = correlation_index[condition]
        axis.set_title(
            f"{condition_labels.get(condition, condition)}\n"
            f"ρ={float(statistic['spearman_rho']):.3f}, "
            f"p={float(statistic['spearman_p']):.3g}"
        )
        axis.set_xlabel(behavior_column)
        axis.set_ylabel("Descending gamma PDC index")
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)

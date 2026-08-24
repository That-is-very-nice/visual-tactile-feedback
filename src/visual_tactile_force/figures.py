"""Publication-ready figures generated from tidy analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

BLUE = "#3F6FB5"
ORANGE = "#D47A2C"
INK = "#20252B"
MID_GREY = "#7C848C"
LIGHT_GREY = "#D9DEE3"


def _p_value_label(value: float) -> str:
    if value < 0.001:
        return f"p = {value:.2e}"
    return f"p = {value:.3f}"


def plot_behavior_figure(
    *,
    summary_table: pd.DataFrame,
    statistics: Mapping[str, object],
    visual_condition: str,
    tactile_condition: str,
    output_paths: Sequence[str | Path],
) -> list[Path]:
    """Plot paired mean-force and CV comparisons for the paper's Figure 3."""

    required = {"subject", "condition", "mean_force", "force_cv"}
    missing = required - set(summary_table.columns)
    if missing:
        raise ValueError(f"Summary table is missing columns: {sorted(missing)}")

    duplicated = summary_table.duplicated(["subject", "condition"])
    if duplicated.any():
        raise ValueError("Summary table contains duplicate subject-condition rows.")

    metric_specs = (
        ("mean_force", "Mean force", "Normalized force", True),
        ("force_cv", "Force variability", "Coefficient of variation", False),
    )
    pivots: dict[str, pd.DataFrame] = {}
    for metric, _, _, _ in metric_specs:
        wide = summary_table.pivot(index="subject", columns="condition", values=metric)
        if visual_condition not in wide or tactile_condition not in wide:
            raise ValueError(f"Both configured conditions are required for {metric}.")
        wide = wide[[visual_condition, tactile_condition]].dropna()
        if len(wide) != summary_table["subject"].nunique():
            raise ValueError(f"Incomplete paired observations for {metric}.")
        pivots[metric] = wide

    figure = Figure(figsize=(7.2, 3.75), facecolor="white", constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)

    for panel_index, (axis, (metric, title, ylabel, focused_scale)) in enumerate(
        zip(axes, metric_specs, strict=True)
    ):
        wide = pivots[metric]
        visual = wide[visual_condition].to_numpy(dtype=float)
        tactile = wide[tactile_condition].to_numpy(dtype=float)
        positions = np.array([0.0, 1.0])

        for visual_value, tactile_value in zip(visual, tactile, strict=True):
            axis.plot(
                positions,
                [visual_value, tactile_value],
                color=MID_GREY,
                alpha=0.48,
                linewidth=0.8,
                zorder=1,
            )

        boxplot = axis.boxplot(
            [visual, tactile],
            positions=positions,
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK, "linewidth": 1.6},
            whiskerprops={"color": MID_GREY, "linewidth": 1.0},
            capprops={"color": MID_GREY, "linewidth": 1.0},
        )
        for patch, color in zip(boxplot["boxes"], (BLUE, ORANGE), strict=True):
            patch.set_facecolor("white")
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)

        axis.scatter(
            np.zeros_like(visual),
            visual,
            s=24,
            marker="o",
            facecolor=BLUE,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
            label="Visual feedback",
        )
        axis.scatter(
            np.ones_like(tactile),
            tactile,
            s=25,
            marker="s",
            facecolor="white",
            edgecolor=ORANGE,
            linewidth=1.2,
            zorder=3,
            label="Tactile feedback",
        )

        stat = statistics[metric]
        if not isinstance(stat, Mapping):
            raise ValueError(f"Statistics entry is not a mapping: {metric}")
        annotation = (
            f"Wilcoxon {_p_value_label(float(stat['p_value']))}; "
            f"$r_z$ = {float(stat['effect_size_r_z']):.3f}"
        )

        axis.set_title(f"({chr(97 + panel_index)}) {title}", loc="left", color=INK, pad=23)
        axis.text(
            0.0,
            1.025,
            annotation,
            transform=axis.transAxes,
            color=INK,
            fontsize=8.5,
            va="bottom",
        )
        axis.set_ylabel(ylabel, color=INK)
        axis.set_xticks(positions, ["Visual\nfeedback", "Tactile\nfeedback"])
        axis.set_xlim(-0.48, 1.48)
        axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(MID_GREY)
        axis.spines["bottom"].set_color(MID_GREY)
        axis.tick_params(colors=INK, labelsize=8.5)

        combined = np.concatenate([visual, tactile])
        data_min = float(np.min(combined))
        data_max = float(np.max(combined))
        data_range = max(data_max - data_min, abs(data_max) * 0.05, 1e-6)
        if focused_scale:
            lower = min(data_min, 1.0) - 0.12 * data_range
            upper = max(data_max, 1.0) + 0.20 * data_range
            axis.set_ylim(lower, upper)
            axis.axhline(1.0, color=MID_GREY, linestyle=(0, (3, 3)), linewidth=0.8)
            axis.text(
                1.46,
                1.0,
                "target",
                color=MID_GREY,
                fontsize=7.5,
                va="bottom",
                ha="right",
            )
        else:
            axis.set_ylim(0.0, data_max + 0.18 * data_range)

    figure.suptitle(
        "Behavioral comparison by feedback condition",
        fontsize=12,
        fontweight="semibold",
        color=INK,
    )
    figure.text(
        0.5,
        -0.025,
        "n = 15 participants; trials 2-5; 10-60 s steady-state window per trial",
        ha="center",
        va="top",
        fontsize=8.5,
        color=MID_GREY,
    )

    written: list[Path] = []
    for output_path in output_paths:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_options: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if path.suffix.lower() == ".png":
            save_options["dpi"] = 300
        figure.savefig(path, **save_options)
        written.append(path)
    return written

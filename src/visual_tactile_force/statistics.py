"""Paired statistical tests used by the paper pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import rankdata, wilcoxon


@dataclass(frozen=True)
class WilcoxonSignedRankResult:
    """Two-sided paired Wilcoxon result with a signed normal-approximation effect size."""

    n_pairs: int
    statistic: float
    p_value: float
    z_statistic: float
    effect_size_r_z: float
    median_difference: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def paired_wilcoxon_signed_rank(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "auto",
) -> WilcoxonSignedRankResult:
    """Compare paired samples using the paper's signed-rank conventions.

    Differences are defined as ``x - y``. NaN pairs and zero differences are
    removed. The p-value is provided by SciPy's two-sided Wilcoxon test, using
    an exact calculation where applicable. ``z_statistic`` is calculated from
    the positive-rank sum without a continuity correction, including the
    standard tie correction. The effect size is ``z / sqrt(n)``.
    """

    x_array = np.asarray(x, dtype=float).reshape(-1)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    if x_array.shape != y_array.shape:
        raise ValueError("Paired samples must have the same shape.")

    finite = np.isfinite(x_array) & np.isfinite(y_array)
    differences = x_array[finite] - y_array[finite]
    differences = differences[differences != 0]
    n_pairs = int(differences.size)

    if n_pairs == 0:
        return WilcoxonSignedRankResult(
            n_pairs=0,
            statistic=0.0,
            p_value=1.0,
            z_statistic=0.0,
            effect_size_r_z=0.0,
            median_difference=0.0,
        )

    scipy_result = wilcoxon(
        differences,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method=method,
    )

    absolute_differences = np.abs(differences)
    ranks = rankdata(absolute_differences, method="average")
    positive_rank_sum = float(ranks[differences > 0].sum())

    mean_rank_sum = n_pairs * (n_pairs + 1) / 4.0
    _, tie_counts = np.unique(absolute_differences, return_counts=True)
    tie_correction = float(np.sum(tie_counts**3 - tie_counts))
    variance = (
        n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 24.0
        - tie_correction / 48.0
    )
    z_statistic = (
        (positive_rank_sum - mean_rank_sum) / np.sqrt(variance)
        if variance > 0
        else 0.0
    )

    return WilcoxonSignedRankResult(
        n_pairs=n_pairs,
        statistic=float(scipy_result.statistic),
        p_value=float(scipy_result.pvalue),
        z_statistic=float(z_statistic),
        effect_size_r_z=float(z_statistic / np.sqrt(n_pairs)),
        median_difference=float(np.median(differences)),
    )


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Adjust a family of p-values with Holm's step-down procedure."""

    values = np.asarray(p_values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and lie between zero and one")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.empty_like(sorted_values)
    running_max = 0.0
    family_size = values.size
    for index, p_value in enumerate(sorted_values):
        candidate = min(1.0, (family_size - index) * p_value)
        running_max = max(running_max, candidate)
        adjusted_sorted[index] = running_max
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted

"""Regression checks for the corrected single-run CMC method baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


CMC_REGRESSION_FIELDS = (
    "subject_count",
    "visual_mean",
    "tactile_mean",
    "n_pairs",
    "statistic",
    "p_value",
    "z_statistic",
    "effect_size_r_z",
    "median_difference",
    "p_value_holm",
)


def compare_cmc_statistics(
    actual: Sequence[Mapping[str, object]],
    expected: Mapping[str, object],
    *,
    absolute_tolerance: float,
) -> dict[str, object]:
    """Compare aggregate CMC statistics without exposing subject-level data."""

    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative")
    expected_bands = expected.get("bands")
    if not isinstance(expected_bands, Mapping):
        raise ValueError("Expected CMC baseline must contain a bands mapping")
    actual_by_band = {str(row["band"]): row for row in actual}

    failures: list[str] = []
    for band, expected_values in expected_bands.items():
        if band not in actual_by_band:
            failures.append(f"missing band: {band}")
            continue
        if not isinstance(expected_values, Mapping):
            failures.append(f"{band}: expected values must be a mapping")
            continue
        actual_values = actual_by_band[band]
        for field in CMC_REGRESSION_FIELDS:
            if field not in expected_values:
                failures.append(f"{band}.{field}: missing expected value")
                continue
            if field not in actual_values:
                failures.append(f"{band}.{field}: missing actual value")
                continue
            difference = abs(
                float(actual_values[field]) - float(expected_values[field])
            )
            if difference > absolute_tolerance:
                failures.append(
                    f"{band}.{field}: difference {difference:.17g} exceeds "
                    f"{absolute_tolerance:.17g}"
                )

    unexpected_bands = sorted(set(actual_by_band) - set(expected_bands))
    failures.extend(f"unexpected band: {band}" for band in unexpected_bands)
    return {
        "status": "pass" if not failures else "fail",
        "absolute_tolerance": absolute_tolerance,
        "failures": failures,
    }

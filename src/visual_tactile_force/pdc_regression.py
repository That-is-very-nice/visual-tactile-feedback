"""Aggregate regression checks for corrected and historical PDC outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


PDC_REGRESSION_FIELDS = (
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


def compare_pdc_statistics(
    actual: Sequence[Mapping[str, object]],
    expected: Mapping[str, object],
    *,
    absolute_tolerance: float,
) -> dict[str, object]:
    """Compare direction/band aggregates without storing subject-level data."""

    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative")
    expected_directions = expected.get("directions")
    if not isinstance(expected_directions, Mapping):
        raise ValueError("Expected PDC baseline must contain a directions mapping")
    actual_by_key = {(str(row["direction"]), str(row["band"])): row for row in actual}
    failures: list[str] = []
    expected_keys: set[tuple[str, str]] = set()
    for direction, bands in expected_directions.items():
        if not isinstance(bands, Mapping):
            failures.append(f"{direction}: expected bands must be a mapping")
            continue
        for band, expected_values in bands.items():
            key = (str(direction), str(band))
            expected_keys.add(key)
            actual_values = actual_by_key.get(key)
            if actual_values is None:
                failures.append(f"missing direction/band: {direction}/{band}")
                continue
            if not isinstance(expected_values, Mapping):
                failures.append(f"{direction}/{band}: expected values must be a mapping")
                continue
            for field in PDC_REGRESSION_FIELDS:
                if field not in expected_values or field not in actual_values:
                    failures.append(f"{direction}/{band}.{field}: missing value")
                    continue
                difference = abs(float(actual_values[field]) - float(expected_values[field]))
                if difference > absolute_tolerance:
                    failures.append(
                        f"{direction}/{band}.{field}: difference {difference:.17g} "
                        f"exceeds {absolute_tolerance:.17g}"
                    )
    for direction, band in sorted(set(actual_by_key) - expected_keys):
        failures.append(f"unexpected direction/band: {direction}/{band}")
    return {
        "status": "pass" if not failures else "fail",
        "absolute_tolerance": absolute_tolerance,
        "failures": failures,
    }

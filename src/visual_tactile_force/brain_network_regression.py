"""Regression checks for the fixed brain-network result."""

from __future__ import annotations

from typing import Mapping, Sequence


def compare_brain_network_significant_rows(
    actual: Sequence[Mapping[str, object]],
    expected: Mapping[str, object],
    *,
    p_column: str,
    absolute_tolerance: float,
) -> dict[str, object]:
    """Compare the canonical significant edge set and its frozen values."""

    expected_rows = expected.get("significant_rows")
    if not isinstance(expected_rows, Sequence):
        raise ValueError("Brain-network baseline must contain significant_rows")
    expected_profile = str(expected.get("analysis_profile", ""))
    profiles = {str(row.get("analysis_profile", "")) for row in actual}
    failures: list[str] = []
    if profiles != {expected_profile}:
        failures.append(
            f"analysis profile differs: actual={sorted(profiles)}, expected={expected_profile!r}"
        )
    expected_canonical_count = expected.get("expected_canonical_row_count")
    if expected_canonical_count is not None and len(actual) != int(expected_canonical_count):
        failures.append(
            "canonical row count differs: "
            f"actual={len(actual)}, expected={int(expected_canonical_count)}"
        )
    actual_within_significant = sum(
        1
        for row in actual
        if str(row.get("edge_scope")) == "within_roi" and float(row[p_column]) < 0.05
    )
    expected_within_value = expected.get("expected_significant_within_roi_count")
    expected_within_significant = (
        int(expected_within_value) if expected_within_value is not None else None
    )
    if (
        expected_within_significant is not None
        and actual_within_significant != expected_within_significant
    ):
        failures.append(
            "significant within-ROI count differs: "
            f"actual={actual_within_significant}, expected={expected_within_significant}"
        )
    actual_rows = [
        row
        for row in actual
        if str(row.get("edge_scope")) == "interregional" and float(row[p_column]) < 0.05
    ]
    key_fields = ("band", "roi_source", "roi_target")
    actual_by_key = {tuple(str(row[field]) for field in key_fields): row for row in actual_rows}
    expected_by_key = {
        tuple(str(row[field]) for field in key_fields): row for row in expected_rows
        if isinstance(row, Mapping)
    }
    if set(actual_by_key) != set(expected_by_key):
        failures.append(
            f"significant edge set differs: actual={sorted(actual_by_key)}, "
            f"expected={sorted(expected_by_key)}"
        )
    for key in sorted(set(actual_by_key) & set(expected_by_key)):
        actual_row = actual_by_key[key]
        expected_row = expected_by_key[key]
        for field in ("mean_difference", p_column):
            if field not in actual_row or field not in expected_row:
                failures.append(f"{key}.{field}: missing value")
                continue
            difference = abs(float(actual_row[field]) - float(expected_row[field]))
            if difference > absolute_tolerance:
                failures.append(
                    f"{key}.{field}: difference {difference:.17g} exceeds "
                    f"{absolute_tolerance:.17g}"
                )
    return {
        "status": "pass" if not failures else "fail",
        "p_column": p_column,
        "absolute_tolerance": absolute_tolerance,
        "actual_significant_count": len(actual_rows),
        "expected_significant_count": len(expected_by_key),
        "actual_significant_within_roi_count": actual_within_significant,
        "expected_significant_within_roi_count": expected_within_significant,
        "failures": failures,
    }

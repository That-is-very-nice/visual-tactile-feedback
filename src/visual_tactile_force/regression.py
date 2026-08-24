"""Regression checks that lock refactored outputs to published results."""

from __future__ import annotations

from typing import Mapping


REGRESSION_FIELDS = ("n_pairs", "statistic", "p_value", "effect_size_r_z")


def compare_behavior_statistics(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    absolute_tolerance: float,
) -> dict[str, object]:
    """Compare behavior results with an aggregate paper baseline."""

    failures: list[str] = []
    comparisons: dict[str, dict[str, object]] = {}
    for metric in ("mean_force", "force_cv"):
        actual_metric = actual.get(metric)
        expected_metric = expected.get(metric)
        if not isinstance(actual_metric, Mapping) or not isinstance(expected_metric, Mapping):
            failures.append(f"Missing metric mapping: {metric}")
            continue
        metric_comparisons: dict[str, object] = {}
        for field in REGRESSION_FIELDS:
            if field not in actual_metric or field not in expected_metric:
                failures.append(f"Missing field: {metric}.{field}")
                continue
            actual_value = float(actual_metric[field])
            expected_value = float(expected_metric[field])
            difference = abs(actual_value - expected_value)
            passed = difference <= absolute_tolerance
            metric_comparisons[field] = {
                "actual": actual_value,
                "expected": expected_value,
                "absolute_difference": difference,
                "pass": passed,
            }
            if not passed:
                failures.append(
                    f"{metric}.{field}: actual={actual_value:.17g}, "
                    f"expected={expected_value:.17g}, difference={difference:.3g}"
                )
        comparisons[metric] = metric_comparisons

    actual_definition = actual.get("difference_definition")
    expected_definition = expected.get("difference_definition")
    if expected_definition is not None and actual_definition != expected_definition:
        failures.append(
            "difference_definition: "
            f"actual={actual_definition!r}, expected={expected_definition!r}"
        )

    return {
        "status": "pass" if not failures else "fail",
        "absolute_tolerance": absolute_tolerance,
        "comparisons": comparisons,
        "failures": failures,
    }

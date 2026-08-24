"""Verify archived PDC Table 2 and Figure 8 values against frozen baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from visual_tactile_force.legacy_pdc import (
    reproduce_published_pdc_behavior_correlations,
    reproduce_published_pdc_statistics,
)
from visual_tactile_force.pdc_regression import compare_pdc_statistics


SUBJECTS = (
    "qh", "wxl", "zys", "lzy", "xl", "zhb", "phoom", "prae",
    "maple", "pepper", "ice", "regina", "ljj", "pathe", "pun",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdc-source", type=Path, required=True)
    parser.add_argument("--correlation-pairs", type=Path, required=True)
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path("configs/pdc_published_legacy_expected.json"),
    )
    args = parser.parse_args(argv)
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    if _sha256(args.pdc_source) != expected["pdc_source_sha256"]:
        raise ValueError("Historical PDC source SHA-256 does not match the baseline")
    if _sha256(args.correlation_pairs) != expected["correlation_pairs_source_sha256"]:
        raise ValueError("Historical correlation-pair SHA-256 does not match the baseline")

    table = reproduce_published_pdc_statistics(
        pd.read_csv(args.pdc_source), subjects=SUBJECTS
    )
    table_report = compare_pdc_statistics(
        table["statistics"],
        expected,
        absolute_tolerance=float(expected["absolute_tolerance"]),
    )
    correlations = reproduce_published_pdc_behavior_correlations(
        pd.read_csv(args.correlation_pairs)
    )
    failures = list(table_report["failures"])
    expected_correlations = expected["figure_8_correlations"]
    for row in correlations["correlations"]:
        frozen = expected_correlations[row["condition"]]
        for field in ("n", "spearman_rho", "spearman_p"):
            difference = abs(float(row[field]) - float(frozen[field]))
            if difference > float(expected["absolute_tolerance"]):
                failures.append(f"{row['condition']}.{field}: difference {difference}")
    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "table_2": table,
        "figure_8": correlations,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

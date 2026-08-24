"""Reproduce the historical Fig. 5/Table 1 aggregate with explicit warnings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from visual_tactile_force.legacy_cmc import reproduce_published_cmc_aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    source_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    report = reproduce_published_cmc_aggregate(
        pd.read_csv(args.input),
        subjects=expected["subjects"],
        metric=expected["metric"],
        expected_rows_per_key=int(expected["expected_rows_per_key"]),
    )

    failures: list[str] = []
    expected_sha256 = expected.get("source_file_sha256")
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        failures.append(
            f"source_file_sha256: actual={source_sha256}, expected={expected_sha256}"
        )
    tolerance = float(expected["absolute_tolerance"])
    expected_by_band = expected["bands"]
    for actual in report["statistics"]:
        band = str(actual["band"])
        for field in (
            "visual_mean",
            "tactile_mean",
            "statistic",
            "p_value",
            "effect_size_r_z",
            "p_value_holm",
        ):
            difference = abs(float(actual[field]) - float(expected_by_band[band][field]))
            if difference > tolerance:
                failures.append(
                    f"{band}.{field}: difference {difference:.17g} exceeds {tolerance:.17g}"
                )
    report["regression"] = {
        "status": "pass" if not failures else "fail",
        "absolute_tolerance": tolerance,
        "expected_file": args.expected.name,
        "failures": failures,
    }
    report["source"] = {
        "file": args.input.name,
        "sha256": source_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

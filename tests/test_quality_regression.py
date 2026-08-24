from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from visual_tactile_force.quality import profile_behavior_dataset
from visual_tactile_force.regression import compare_behavior_statistics


def _write_force_file(path: Path, measured: float) -> None:
    rows = []
    for trial in range(1, 6):
        base = trial * 100_000.0
        rows.append(["start", base, 10.0, measured])
        for step in range(1, 60):
            rows.append(["", base + step * 1_000.0, 10.0, measured])
        rows.append(["end", base + 60_000.0, 10.0, measured])
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


class BehaviorQualityTests(unittest.TestCase):
    def test_complete_dataset_passes_and_reports_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_force_file(root / "s1_st_no.csv", 10.0)
            _write_force_file(root / "s1_st_tf2.csv", 11.0)
            _write_force_file(root / "unused.csv", 12.0)

            report = profile_behavior_dataset(
                data_dir=root,
                subjects=["s1"],
                conditions=["st_no", "st_tf2"],
                trial_indices=[2, 3, 4, 5],
                start_offsets_ms={"st_no": 10_000.0, "st_tf2": 10_000.0},
                expected_sampling_rate_hz=1.0,
                compute_sha256=True,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["checked_file_count"], 2)
            self.assertEqual(report["extra_csv_files"], ["unused.csv"])
            self.assertEqual(len(report["files"][0]["sha256"]), 64)

    def test_missing_file_fails_with_actionable_issue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = profile_behavior_dataset(
                data_dir=directory,
                subjects=["s1"],
                conditions=["st_no"],
                trial_indices=[2],
                start_offsets_ms={"st_no": 10_000.0},
            )
            self.assertEqual(report["status"], "fail")
            self.assertIn("Missing configured file", report["issues"][0])


class PaperRegressionTests(unittest.TestCase):
    def test_exact_aggregate_baseline_passes(self) -> None:
        actual = {
            "difference_definition": "st_no minus st_tf2",
            "mean_force": {
                "n_pairs": 15,
                "statistic": 39.0,
                "p_value": 0.25238037109375,
                "effect_size_r_z": -0.3079589415448419,
            },
            "force_cv": {
                "n_pairs": 15,
                "statistic": 0.0,
                "p_value": 0.00006103515625,
                "effect_size_r_z": -0.8798826901281197,
            },
        }
        report = compare_behavior_statistics(
            actual,
            actual,
            absolute_tolerance=1e-12,
        )
        self.assertEqual(report["status"], "pass")

    def test_changed_statistic_fails(self) -> None:
        expected = {
            "mean_force": {
                field: 1.0
                for field in ("n_pairs", "statistic", "p_value", "effect_size_r_z")
            },
            "force_cv": {
                field: 1.0
                for field in ("n_pairs", "statistic", "p_value", "effect_size_r_z")
            },
        }
        actual = {
            "mean_force": dict(expected["mean_force"]),
            "force_cv": dict(expected["force_cv"]),
        }
        actual["force_cv"]["statistic"] = 2.0
        report = compare_behavior_statistics(
            actual,
            expected,
            absolute_tolerance=1e-12,
        )
        self.assertEqual(report["status"], "fail")
        self.assertIn("force_cv.statistic", report["failures"][0])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from visual_tactile_force.behavior import (
    FORCE_COLUMNS,
    analyze_force_dataset,
    compute_trial_metrics,
    read_force_csv,
    sampling_rate_hz,
    trial_windows,
)


class BehaviorUnitTests(unittest.TestCase):
    def test_sampling_rate_uses_millisecond_timestamps(self) -> None:
        self.assertAlmostEqual(sampling_rate_hz(np.array([0.0, 10.0, 20.0])), 100.0)

    def test_trial_window_applies_steady_state_offset(self) -> None:
        frame = pd.DataFrame(
            [
                ["start", 0.0, 10.0, 10.0],
                ["", 10_000.0, 10.0, 10.0],
                ["end", 60_000.0, 10.0, 10.0],
            ],
            columns=FORCE_COLUMNS,
        )
        windows = trial_windows(frame, start_offset_ms=10_000.0)
        self.assertEqual(windows[0].start_ms, 10_000.0)
        self.assertEqual(windows[0].end_ms, 60_000.0)

    def test_reader_does_not_shift_columns_after_a_corrupt_trailing_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            path.write_text(
                "none,0,bad,14,10,0\nstart,100,10,9\nend,200,10,11\n",
                encoding="utf-8",
            )
            frame = read_force_csv(path)
            self.assertEqual(frame.loc[1, "event"], "start")
            self.assertEqual(frame.loc[1, "time_ms"], 100.0)

    def test_reader_drops_non_marker_rows_with_invalid_times(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            path.write_text(
                "none,bad,10,9\nstart,100,10,9\nend,200,10,11\n",
                encoding="utf-8",
            )
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                frame = read_force_csv(path)
            self.assertEqual(len(frame), 2)
            self.assertEqual(len(captured), 1)

    def test_trial_metrics_match_sample_cv_definition(self) -> None:
        result = compute_trial_metrics(
            subject="s1",
            condition="visual",
            trial_index=2,
            time_ms=np.array([0.0, 10.0, 20.0]),
            target_force=np.array([10.0, 10.0, 10.0]),
            measured_force=np.array([8.0, 10.0, 12.0]),
            lowpass_hz=None,
            normalization="per_target_median",
            ddof=1,
        )
        self.assertAlmostEqual(result.mean_force, 1.0)
        self.assertAlmostEqual(result.force_standard_deviation, 0.2)
        self.assertAlmostEqual(result.force_cv, 0.2)

    def test_dataset_analysis_is_strict_and_subject_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition, measured in (("st_no", 10.0), ("st_tf2", 12.0)):
                rows = []
                for trial in range(1, 6):
                    base = trial * 100_000.0
                    rows.append(["start", base, 10.0, measured])
                    for step in range(1, 60):
                        rows.append(["", base + step * 1_000.0, 10.0, measured])
                    rows.append(["end", base + 60_000.0, 10.0, measured])
                pd.DataFrame(rows).to_csv(root / f"s1_{condition}.csv", header=False, index=False)

            trial_table, summary_table = analyze_force_dataset(
                data_dir=root,
                subjects=["s1"],
                conditions=["st_no", "st_tf2"],
                trial_indices=[2, 3, 4, 5],
                start_offsets_ms={"st_no": 10_000.0, "st_tf2": 10_000.0},
                lowpass_hz=None,
            )
            self.assertEqual(len(trial_table), 8)
            self.assertEqual(len(summary_table), 2)
            tactile = summary_table.loc[summary_table["condition"] == "st_tf2"].iloc[0]
            self.assertAlmostEqual(tactile["mean_force"], 1.2)
            self.assertEqual(tactile["n_trials"], 4)

    def test_subject_directory_override_handles_legacy_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_dir = root / "shared_folder"
            legacy_dir.mkdir()
            rows = []
            for trial in range(1, 6):
                base = trial * 100_000.0
                rows.append(["start", base, 10.0, 10.0])
                for step in range(1, 60):
                    rows.append(["", base + step * 1_000.0, 10.0, 10.0])
                rows.append(["end", base + 60_000.0, 10.0, 10.0])
            pd.DataFrame(rows).to_csv(
                legacy_dir / "pathe_st_no.csv",
                header=False,
                index=False,
            )

            trial_table, _ = analyze_force_dataset(
                data_dir=root,
                subjects=["pathe"],
                conditions=["st_no"],
                trial_indices=[2],
                start_offsets_ms={"st_no": 10_000.0},
                file_template="{subject_dir}/{subject}_{condition}.csv",
                subject_subdirectories={"pathe": "shared_folder"},
                lowpass_hz=None,
            )
            self.assertEqual(len(trial_table), 1)


if __name__ == "__main__":
    unittest.main()

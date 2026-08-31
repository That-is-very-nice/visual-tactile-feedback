from __future__ import annotations

import unittest

import pandas as pd

from visual_tactile_force.pdc import PAPER_PDC_METRIC

from visual_tactile_force.pdc_batch import (
    assert_unique_pdc_rows,
    correlate_pdc_with_behavior,
    paper_pdc_results_to_frame,
    summarize_paper_pdc,
)


class PdcBatchTests(unittest.TestCase):
    def _results(self) -> list[dict[str, object]]:
        results = []
        for subject_index in range(1, 6):
            for condition, shift in (("st_no", 0.0), ("st_tf2", 0.2)):
                results.append(
                    {
                        "subject": f"s{subject_index}",
                        "condition": condition,
                        "event_code": 1 if condition == "st_no" else 9,
                        "summary_metric": PAPER_PDC_METRIC,
                        "pdc_by_direction_band": {
                            direction: {
                                band: subject_index + shift + band_index
                                for band_index, band in enumerate(("alpha", "beta", "gamma"))
                            }
                            for direction in ("descending", "ascending")
                        },
                    }
                )
        return results

    def test_tidy_frame_and_statistics_cover_both_directions(self) -> None:
        frame = paper_pdc_results_to_frame(self._results())
        statistics = summarize_paper_pdc(
            frame, visual_condition="st_no", tactile_condition="st_tf2"
        )
        self.assertEqual(len(frame), 5 * 2 * 2 * 3)
        self.assertEqual(len(statistics), 6)
        self.assertTrue(all(row["subject_count"] == 5 for row in statistics))
        self.assertTrue(all("p_value_holm" in row for row in statistics))

    def test_duplicate_keys_are_rejected(self) -> None:
        frame = paper_pdc_results_to_frame(self._results())
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            assert_unique_pdc_rows(duplicate)

    def test_corrected_behavior_correlation_uses_subject_condition_pairs(self) -> None:
        frame = paper_pdc_results_to_frame(self._results())
        behavior = pd.DataFrame(
            [
                {"subject": f"s{subject}", "condition": condition, "force_cv": subject / 100}
                for subject in range(1, 6)
                for condition in ("st_no", "st_tf2")
            ]
        )
        correlations = correlate_pdc_with_behavior(frame, behavior)
        self.assertEqual(len(correlations), 2)
        self.assertTrue(
            all(abs(float(row["spearman_rho"]) - 1.0) < 1e-12 for row in correlations)
        )


if __name__ == "__main__":
    unittest.main()

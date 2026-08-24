from __future__ import annotations

import unittest

import pandas as pd

from visual_tactile_force.cmc_batch import (
    CMC_CANDIDATE_METRICS,
    assert_unique_cmc_rows,
    cmc_results_to_frame,
    paper_cmc_results_to_frame,
    summarize_paper_cmc,
    summarize_cmc_candidates,
)


class CmcBatchTests(unittest.TestCase):
    def test_stable_output_contains_only_the_frozen_metric(self) -> None:
        results = []
        for subject_index in range(1, 5):
            for condition, shift in (("st_no", 0.0), ("st_tf2", 0.1)):
                results.append(
                    {
                        "subject": f"s{subject_index}",
                        "condition": condition,
                        "event_code": 1 if condition == "st_no" else 9,
                        "summary_metric": "max_mean_suprathreshold_excess",
                        "cmc_by_band": {
                            band: subject_index + band_index + shift
                            for band_index, band in enumerate(
                                ("alpha", "beta", "gamma")
                            )
                        },
                    }
                )
        frame = paper_cmc_results_to_frame(results)
        statistics = summarize_paper_cmc(
            frame,
            visual_condition="st_no",
            tactile_condition="st_tf2",
        )

        self.assertEqual(len(frame), 24)
        self.assertEqual(set(frame.columns), {
            "subject", "condition", "event_code", "band", "cmc_index", "metric"
        })
        self.assertEqual(len(statistics), 3)
        self.assertTrue(all(row["subject_count"] == 4 for row in statistics))
        self.assertTrue(all("p_value_holm" in row for row in statistics))

    def test_stable_output_rejects_a_different_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected"):
            paper_cmc_results_to_frame(
                [
                    {
                        "subject": "s1",
                        "condition": "st_no",
                        "event_code": 1,
                        "summary_metric": "max_band_mean",
                        "cmc_by_band": {"alpha": 0.1},
                    }
                ]
            )

    def test_tidy_conversion_and_statistics_cover_three_bands(self) -> None:
        results = []
        for subject_index in range(1, 5):
            for condition, shift in (("st_no", 0.0), ("st_tf2", 0.1)):
                results.append(
                    {
                        "subject": f"s{subject_index}",
                        "condition": condition,
                        "event_code": 1 if condition == "st_no" else 9,
                        "roi_candidate_metrics": {
                            band: {
                                metric: subject_index + shift + metric_index
                                for metric_index, metric in enumerate(CMC_CANDIDATE_METRICS)
                            }
                            for band in ("alpha", "beta", "gamma")
                        },
                    }
                )
        frame = cmc_results_to_frame(results)
        statistics = summarize_cmc_candidates(
            frame,
            visual_condition="st_no",
            tactile_condition="st_tf2",
        )
        self.assertEqual(len(frame), 24)
        self.assertEqual(len(statistics), 3 * len(CMC_CANDIDATE_METRICS))
        self.assertTrue(all("p_value_holm" in row for row in statistics))

    def test_duplicate_analysis_units_are_rejected(self) -> None:
        results = []
        for condition in ("st_no", "st_tf2"):
            results.append(
                {
                    "subject": "s1",
                    "condition": condition,
                    "event_code": 1,
                    "roi_candidate_metrics": {
                        band: {metric: 0.1 for metric in CMC_CANDIDATE_METRICS}
                        for band in ("alpha", "beta", "gamma")
                    },
                }
            )
        frame = cmc_results_to_frame(results)
        duplicated = frame.iloc[[0]]
        frame = pd.concat([frame, duplicated], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            assert_unique_cmc_rows(frame)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            summarize_cmc_candidates(
                frame,
                visual_condition="st_no",
                tactile_condition="st_tf2",
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from visual_tactile_force.cmc import (
    coherence_confidence_limit,
    segment_trials,
    summarize_roi_cmc,
    validate_aligned_segments,
)


class SegmentationTests(unittest.TestCase):
    def test_paper_window_produces_twenty_segments_and_excludes_trial_one(self) -> None:
        data = np.empty((5, 2, 60_000), dtype=np.int16)
        for trial in range(5):
            data[trial] = trial + 1

        segments, windows = segment_trials(
            data,
            sampling_rate_hz=1_000.0,
            trial_indices=[2, 3, 4, 5],
            tmin_s=10.0,
            tmax_s=60.0,
            segment_duration_s=10.0,
        )

        self.assertEqual(segments.shape, (20, 2, 10_000))
        self.assertEqual([window.trial_index for window in windows[:5]], [2] * 5)
        self.assertEqual((windows[0].start_s, windows[0].stop_s), (10.0, 20.0))
        self.assertEqual((windows[-1].start_s, windows[-1].stop_s), (50.0, 60.0))
        np.testing.assert_array_equal(np.unique(segments[:, 0, 0]), [2, 3, 4, 5])

    def test_non_integral_segment_layout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "whole number"):
            segment_trials(
                np.zeros((5, 60)),
                sampling_rate_hz=1.0,
                trial_indices=[2, 3, 4, 5],
                tmin_s=10.0,
                tmax_s=60.0,
                segment_duration_s=12.0,
            )

    def test_alignment_checks_segment_and_sample_dimensions(self) -> None:
        validate_aligned_segments(np.zeros((20, 61, 100)), np.zeros((20, 1, 100)))
        with self.assertRaisesRegex(ValueError, "segment counts"):
            validate_aligned_segments(np.zeros((20, 61, 100)), np.zeros((19, 1, 100)))


class CmcSummaryTests(unittest.TestCase):
    def test_confidence_limit_matches_paper_formula(self) -> None:
        actual = coherence_confidence_limit(alpha=0.05, independent_segment_count=20)
        expected = 1.0 - 0.05 ** (1.0 / 19.0)
        self.assertAlmostEqual(actual, expected)

    def test_roi_summary_uses_only_requested_channels(self) -> None:
        frequencies = np.array([8.0, 10.0, 13.0, 20.0])
        coherence = np.array(
            [
                [0.20, 0.30, 0.10, 0.40],
                [0.90, 0.90, 0.90, 0.90],
                [0.15, 0.25, 0.35, 0.20],
            ]
        )
        result = summarize_roi_cmc(
            coherence,
            frequencies,
            channel_names=["C3", "O2", "Cz"],
            roi_channels=["C3", "Cz"],
            bands_hz={"alpha": [8.0, 13.0], "beta": [13.0, 30.0]},
            confidence_limit=0.20,
        )

        self.assertAlmostEqual(
            result["alpha"]["max_mean_suprathreshold_excess"], 0.10
        )
        self.assertAlmostEqual(result["alpha"]["max_suprathreshold_excess"], 0.15)
        self.assertAlmostEqual(
            result["alpha"]["max_normalized_suprathreshold_area"], 0.10
        )
        self.assertAlmostEqual(result["beta"]["max_band_mean"], 0.275)


if __name__ == "__main__":
    unittest.main()

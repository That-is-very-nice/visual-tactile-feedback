from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from visual_tactile_force.brain_network import PAPER_BANDS_HZ, PAPER_ROI_CHANNELS
from visual_tactile_force.brain_network_batch import (
    assert_unique_brain_network_rows,
    exact_studentized_max_t,
    summarize_declared_holm,
    summarize_published_style_max_t,
)


class BrainNetworkBatchTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = []
        rois = list(PAPER_ROI_CHANNELS)
        for subject_index in range(6):
            for condition, condition_shift in (("st_no", 0.0), ("st_tf2", 0.02)):
                for band_index, band in enumerate(PAPER_BANDS_HZ):
                    for source_index, source in enumerate(rois):
                        for target in rois[source_index:]:
                            rows.append(
                                {
                                    "subject": f"s{subject_index}",
                                    "condition": condition,
                                    "band": band,
                                    "roi_source": source,
                                    "roi_target": target,
                                    "edge_scope": (
                                        "within_roi" if source == target else "interregional"
                                    ),
                                    "absolute_imaginary_coherence": (
                                        subject_index / 100 + band_index / 100 + condition_shift
                                    ),
                                }
                            )
        return pd.DataFrame(rows)

    def test_declared_holm_has_one_global_225_test_family(self) -> None:
        rows = summarize_declared_holm(
            self._frame(), visual_condition="st_no", tactile_condition="st_tf2"
        )
        self.assertEqual(len(rows), 225)
        self.assertTrue(all(row["correction_family_size"] == 225 for row in rows))

    def test_published_style_max_t_uses_55_tests_per_band(self) -> None:
        rows = summarize_published_style_max_t(
            self._frame(), visual_condition="st_no", tactile_condition="st_tf2"
        )
        self.assertEqual(len(rows), 275)
        self.assertTrue(all(row["correction_family_size"] == 55 for row in rows))
        self.assertTrue(all(row["permutation_count"] == 64 for row in rows))

    def test_exact_sign_flip_is_deterministic_and_bounded(self) -> None:
        differences = np.array(
            [[1.0, -0.2], [1.1, 0.1], [0.9, -0.1], [1.2, 0.2], [0.8, -0.2]]
        )
        first = exact_studentized_max_t(differences)
        second = exact_studentized_max_t(differences)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(np.all((first[1] >= 0) & (first[1] <= 1)))

    def test_duplicate_subject_edge_is_rejected(self) -> None:
        frame = self._frame()
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            assert_unique_brain_network_rows(duplicate)


if __name__ == "__main__":
    unittest.main()

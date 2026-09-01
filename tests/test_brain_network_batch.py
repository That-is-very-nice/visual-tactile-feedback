from __future__ import annotations

import unittest

import pandas as pd

from visual_tactile_force.brain_network import PAPER_BANDS_HZ, PAPER_ROI_CHANNELS
from visual_tactile_force.brain_network_batch import (
    assert_unique_brain_network_rows,
    collapse_directed_brain_network_rows,
    summarize_brain_network_wilcoxon_holm,
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

    def test_each_band_has_100_directed_tests(self) -> None:
        rows = summarize_brain_network_wilcoxon_holm(
            self._frame(), visual_condition="st_no", tactile_condition="st_tf2"
        )
        self.assertEqual(len(rows), 500)
        self.assertTrue(all(row["correction_family_size"] == 100 for row in rows))
        self.assertTrue(all("p_value_holm" in row for row in rows))

    def test_directed_rows_collapse_to_55_pairs_per_band(self) -> None:
        rows = summarize_brain_network_wilcoxon_holm(
            self._frame(), visual_condition="st_no", tactile_condition="st_tf2"
        )
        collapsed = collapse_directed_brain_network_rows(rows)
        self.assertEqual(len(collapsed), 275)
        self.assertEqual(
            sum(row["edge_scope"] == "interregional" for row in collapsed),
            225,
        )
        self.assertEqual(
            sum(row["edge_scope"] == "within_roi" for row in collapsed),
            50,
        )

    def test_duplicate_subject_edge_is_rejected(self) -> None:
        frame = self._frame()
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            assert_unique_brain_network_rows(duplicate)


if __name__ == "__main__":
    unittest.main()

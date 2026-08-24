from __future__ import annotations

import unittest

import pandas as pd

from visual_tactile_force.legacy_cmc import reproduce_published_cmc_aggregate


class LegacyCmcTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = []
        for subject_index, subject in enumerate(("s1", "s2", "s3"), start=1):
            for event, shift in (("st_no", 0.0), ("st_tf2", 0.2)):
                for band_index, band in enumerate(("alpha", "beta", "gamma")):
                    for run_shift in (0.0, 1.0):
                        rows.append(
                            {
                                "subject": subject,
                                "event": event,
                                "band": band,
                                "cmc5": subject_index
                                + band_index
                                + shift
                                + run_shift,
                            }
                        )
        return pd.DataFrame(rows)

    def test_historical_adapter_makes_duplicate_mean_explicit(self) -> None:
        report = reproduce_published_cmc_aggregate(
            self._frame(), subjects=("s1", "s2", "s3")
        )
        self.assertEqual(report["status"], "historical_published_output_reproduced")
        self.assertEqual(report["source_row_count"], 36)
        self.assertEqual(report["unique_key_count"], 18)
        self.assertEqual(report["conflicting_key_count"], 18)
        self.assertEqual(len(report["statistics"]), 3)
        self.assertAlmostEqual(report["statistics"][0]["visual_mean"], 2.5)
        self.assertAlmostEqual(report["statistics"][0]["tactile_mean"], 2.7)

    def test_historical_adapter_rejects_nonduplicated_input(self) -> None:
        frame = self._frame().groupby(
            ["subject", "event", "band"], as_index=False
        )["cmc5"].mean()
        with self.assertRaisesRegex(ValueError, "Expected 2 rows"):
            reproduce_published_cmc_aggregate(frame)


if __name__ == "__main__":
    unittest.main()

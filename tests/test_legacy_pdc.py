from __future__ import annotations

import unittest

import pandas as pd

from visual_tactile_force.legacy_pdc import (
    reproduce_published_pdc_behavior_correlations,
    reproduce_published_pdc_statistics,
)


class LegacyPdcTests(unittest.TestCase):
    def test_statistics_require_unique_historical_keys(self) -> None:
        rows = []
        for subject in ("s1", "s2", "s3", "s4"):
            for event, shift in (("st_no", 0.0), ("st_tf2", 0.1)):
                for band_index, band in enumerate(("alpha", "beta", "gamma")):
                    rows.append(
                        {
                            "subject": subject,
                            "event": event,
                            "band": band,
                            "pdc_down5": band_index + shift,
                            "pdc_up5": band_index - shift,
                        }
                    )
        frame = pd.DataFrame(rows)
        result = reproduce_published_pdc_statistics(frame)
        self.assertEqual(len(result["statistics"]), 6)
        duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            reproduce_published_pdc_statistics(duplicated)

    def test_correlation_adapter_recomputes_archived_pair_table(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "subject": f"s{index}",
                    "event": event,
                    "band": "gamma",
                    "pdc_down5": float(index),
                    "fluct_across_trials": float(6 - index),
                }
                for event in ("st_no", "st_tf2")
                for index in range(1, 6)
            ]
        )
        result = reproduce_published_pdc_behavior_correlations(frame)
        self.assertTrue(
            all(
                abs(float(row["spearman_rho"]) + 1.0) < 1e-12
                for row in result["correlations"]
            )
        )


if __name__ == "__main__":
    unittest.main()

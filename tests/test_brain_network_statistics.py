from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from visual_tactile_force.brain_network import PAPER_ROI_CHANNELS
from visual_tactile_force.brain_network_statistics import (
    load_saved_brain_network_statistics,
)


class BrainNetworkStatisticsTests(unittest.TestCase):
    def test_saved_columns_are_normalized_to_holm_schema(self) -> None:
        rows = []
        rois = list(PAPER_ROI_CHANNELS)
        for band in ("delta", "theta", "alpha", "beta", "gamma"):
            for source in rois:
                for target in rois:
                    rows.append(
                        {
                            "event_group": "fb",
                            "frequency": band,
                            "comparison": "st_no vs st_tf2",
                            "source": source,
                            "target": target,
                            "mean_diff": -0.02,
                            "p_maxT": 0.03,
                        }
                    )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "statistics.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            result = load_saved_brain_network_statistics(path)
        self.assertEqual(len(result), 500)
        self.assertIn("p_value_holm", result.columns)
        self.assertNotIn("p_value_max_t", result.columns)
        self.assertTrue(all(result["correction_family_size"] == 100))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from visual_tactile_force.legacy_brain_network import load_published_max_t


class LegacyBrainNetworkTests(unittest.TestCase):
    def test_directed_rows_collapse_to_one_undirected_edge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "max_t.csv"
            pd.DataFrame(
                [
                    {
                        "event_group": "fb",
                        "frequency": "theta",
                        "comparison": "st_no vs st_tf2",
                        "source": "Left_Frontal",
                        "target": "Left_Central",
                        "mean_diff": -0.02,
                        "p_maxT": 0.03,
                    },
                    {
                        "event_group": "fb",
                        "frequency": "theta",
                        "comparison": "st_no vs st_tf2",
                        "source": "Left_Central",
                        "target": "Left_Frontal",
                        "mean_diff": -0.02,
                        "p_maxT": 0.03,
                    },
                ]
            ).to_csv(path, index=False)
            result = load_published_max_t(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["directed_rows_collapsed"], 2)
        self.assertEqual(result.iloc[0]["roi_source"], "Left_Frontal")
        self.assertEqual(result.iloc[0]["roi_target"], "Left_Central")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from visual_tactile_force.brain_network_regression import (
    compare_brain_network_significant_rows,
)


class BrainNetworkRegressionTests(unittest.TestCase):
    def test_significant_edge_and_values_are_frozen(self) -> None:
        row = {
            "band": "theta",
            "roi_source": "Left_Frontal",
            "roi_target": "Left_Central",
            "edge_scope": "interregional",
            "mean_difference": -0.02,
            "analysis_profile": "wilcoxon_holm_per_band_directed_100",
            "p_value_holm": 0.03,
        }
        report = compare_brain_network_significant_rows(
            [row],
            {
                "analysis_profile": "wilcoxon_holm_per_band_directed_100",
                "significant_rows": [dict(row)],
            },
            p_column="p_value_holm",
            absolute_tolerance=1e-12,
        )
        self.assertEqual(report["status"], "pass")


if __name__ == "__main__":
    unittest.main()

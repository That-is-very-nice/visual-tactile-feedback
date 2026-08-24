from __future__ import annotations

import copy
import unittest

from visual_tactile_force.cmc_regression import compare_cmc_statistics


class CmcRegressionTests(unittest.TestCase):
    def _actual(self) -> list[dict[str, object]]:
        rows = []
        for index, band in enumerate(("alpha", "beta", "gamma"), start=1):
            rows.append(
                {
                    "band": band,
                    "subject_count": 15,
                    "visual_mean": index + 0.1,
                    "tactile_mean": index + 0.2,
                    "n_pairs": 15,
                    "statistic": float(index),
                    "p_value": 0.5,
                    "z_statistic": -0.2,
                    "effect_size_r_z": -0.1,
                    "median_difference": -0.05,
                    "p_value_holm": 1.0,
                }
            )
        return rows

    def _expected(self, actual: list[dict[str, object]]) -> dict[str, object]:
        return {
            "bands": {
                str(row["band"]): {
                    key: value for key, value in row.items() if key != "band"
                }
                for row in actual
            }
        }

    def test_exact_aggregate_baseline_passes(self) -> None:
        actual = self._actual()
        report = compare_cmc_statistics(
            actual, self._expected(actual), absolute_tolerance=1e-12
        )
        self.assertEqual(report["status"], "pass")

    def test_changed_statistic_fails(self) -> None:
        actual = self._actual()
        expected = copy.deepcopy(self._expected(actual))
        expected["bands"]["beta"]["effect_size_r_z"] = -0.3
        report = compare_cmc_statistics(
            actual, expected, absolute_tolerance=1e-12
        )
        self.assertEqual(report["status"], "fail")
        self.assertIn("beta.effect_size_r_z", report["failures"][0])


if __name__ == "__main__":
    unittest.main()

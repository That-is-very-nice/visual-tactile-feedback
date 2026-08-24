from __future__ import annotations

import unittest

from visual_tactile_force.pdc_regression import PDC_REGRESSION_FIELDS, compare_pdc_statistics


class PdcRegressionTests(unittest.TestCase):
    def test_exact_statistics_pass_and_changed_value_fails(self) -> None:
        row = {"direction": "descending", "band": "gamma"}
        row.update({field: 1.0 for field in PDC_REGRESSION_FIELDS})
        expected = {
            "directions": {
                "descending": {
                    "gamma": {field: 1.0 for field in PDC_REGRESSION_FIELDS}
                }
            }
        }
        self.assertEqual(
            compare_pdc_statistics([row], expected, absolute_tolerance=0.0)["status"],
            "pass",
        )
        row["p_value"] = 0.5
        self.assertEqual(
            compare_pdc_statistics([row], expected, absolute_tolerance=1e-12)["status"],
            "fail",
        )


if __name__ == "__main__":
    unittest.main()

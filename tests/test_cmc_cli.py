from __future__ import annotations

import unittest

from visual_tactile_force.cmc_cli import _validate_cmc_config


class CmcCliConfigTests(unittest.TestCase):
    def _config(self) -> dict[str, object]:
        return {
            "trial_indices": [2, 3, 4, 5],
            "tmin_s": 10.0,
            "tmax_s": 60.0,
            "segment_duration_s": 10.0,
            "expected_segment_count": 20,
            "emg_filter_after_alignment": "none",
            "emg_rectification": False,
            "estimator": "multitaper magnitude-squared coherence",
            "apply_csd": True,
            "summary_metric": "max_mean_suprathreshold_excess",
        }

    def test_paper_configuration_passes(self) -> None:
        _validate_cmc_config(self._config())

    def test_wrong_metric_is_rejected(self) -> None:
        config = self._config()
        config["summary_metric"] = "max_band_mean"
        with self.assertRaisesRegex(ValueError, "summary metric"):
            _validate_cmc_config(config)

    def test_inconsistent_segment_count_is_rejected(self) -> None:
        config = self._config()
        config["expected_segment_count"] = 16
        with self.assertRaisesRegex(ValueError, "expected_segment_count"):
            _validate_cmc_config(config)


if __name__ == "__main__":
    unittest.main()

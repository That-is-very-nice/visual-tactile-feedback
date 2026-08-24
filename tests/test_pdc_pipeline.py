from __future__ import annotations

import unittest

import numpy as np

from visual_tactile_force.pdc_pipeline import resample_segments


class PdcPipelineTests(unittest.TestCase):
    def test_paper_segment_resamples_from_1000_to_160_hz(self) -> None:
        data = np.zeros((20, 2, 10_000))
        result = resample_segments(data, source_rate_hz=1000.0, target_rate_hz=160.0)
        self.assertEqual(result.shape, (20, 2, 1600))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from visual_tactile_force.cmc_figures import plot_cmc_summary


class CmcFigureTests(unittest.TestCase):
    def test_cmc_figure_exports_nonempty_png(self) -> None:
        rows = []
        for subject_index in range(4):
            for band_index, band in enumerate(("alpha", "beta", "gamma")):
                rows.extend(
                    [
                        {
                            "subject": f"s{subject_index}",
                            "condition": "st_no",
                            "band": band,
                            "cmc_index": 0.01 * (subject_index + band_index + 1),
                        },
                        {
                            "subject": f"s{subject_index}",
                            "condition": "st_tf2",
                            "band": band,
                            "cmc_index": 0.012 * (subject_index + band_index + 1),
                        },
                    ]
                )
        statistics = [
            {"band": band, "p_value_holm": 1.0}
            for band in ("alpha", "beta", "gamma")
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cmc.png"
            plot_cmc_summary(
                pd.DataFrame(rows),
                statistics,
                visual_condition="st_no",
                tactile_condition="st_tf2",
                output_paths=(output,),
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()

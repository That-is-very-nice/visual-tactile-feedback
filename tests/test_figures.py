from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from visual_tactile_force.figures import plot_behavior_figure


class BehaviorFigureTests(unittest.TestCase):
    def test_behavior_figure_exports_nonempty_png(self) -> None:
        summary = pd.DataFrame(
            [
                {"subject": subject, "condition": condition, "mean_force": mean, "force_cv": cv}
                for subject, values in {
                    "s1": ((1.00, 0.10), (1.03, 0.16)),
                    "s2": ((0.98, 0.09), (1.01, 0.15)),
                    "s3": ((1.05, 0.11), (1.08, 0.18)),
                }.items()
                for condition, (mean, cv) in zip(("st_no", "st_tf2"), values, strict=True)
            ]
        )
        statistics = {
            "mean_force": {"p_value": 0.25, "effect_size_r_z": -0.31},
            "force_cv": {"p_value": 0.000061, "effect_size_r_z": -0.88},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "figure.png"
            written = plot_behavior_figure(
                summary_table=summary,
                statistics=statistics,
                visual_condition="st_no",
                tactile_condition="st_tf2",
                output_paths=[output],
            )
            self.assertEqual(written, [output])
            self.assertGreater(output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from visual_tactile_force.statistics import holm_adjust, paired_wilcoxon_signed_rank


class StatisticsUnitTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_in_sorted_p_values(self) -> None:
        adjusted = holm_adjust(np.array([0.04, 0.01, 0.03]))
        np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06])

    def test_all_negative_differences_match_paper_effect_convention(self) -> None:
        visual = np.arange(15, dtype=float)
        tactile = visual + np.arange(1.0, 16.0)
        result = paired_wilcoxon_signed_rank(visual, tactile)

        self.assertEqual(result.n_pairs, 15)
        self.assertEqual(result.statistic, 0.0)
        self.assertAlmostEqual(result.p_value, 0.00006103515625)
        self.assertAlmostEqual(result.effect_size_r_z, -0.8798826901281197)
        self.assertEqual(result.median_difference, -8.0)

    def test_rank_sum_39_matches_paper_mean_force_reporting(self) -> None:
        ranks = np.arange(1.0, 16.0)
        positive_ranks = {1, 2, 3, 4, 14, 15}
        differences = np.array(
            [rank if int(rank) in positive_ranks else -rank for rank in ranks]
        )
        result = paired_wilcoxon_signed_rank(differences, np.zeros_like(differences))

        self.assertEqual(result.statistic, 39.0)
        self.assertAlmostEqual(result.p_value, 0.25238037109375)
        self.assertAlmostEqual(result.effect_size_r_z, -0.3079589415448419)

    def test_nan_pairs_and_zero_differences_are_excluded(self) -> None:
        result = paired_wilcoxon_signed_rank(
            np.array([1.0, 2.0, np.nan, 4.0]),
            np.array([1.0, 3.0, 2.0, 6.0]),
        )
        self.assertEqual(result.n_pairs, 2)
        self.assertLess(result.effect_size_r_z, 0)

    def test_all_zero_differences_return_neutral_result(self) -> None:
        result = paired_wilcoxon_signed_rank(np.ones(3), np.ones(3))
        self.assertEqual(result.n_pairs, 0)
        self.assertEqual(result.p_value, 1.0)
        self.assertEqual(result.effect_size_r_z, 0.0)


if __name__ == "__main__":
    unittest.main()

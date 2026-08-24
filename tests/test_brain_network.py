from __future__ import annotations

import unittest

import numpy as np

from visual_tactile_force.brain_network import (
    PAPER_BANDS_HZ,
    PAPER_ROI_CHANNELS,
    aggregate_roi_connectivity,
    build_roi_channel_edges,
    mapped_channels,
    validate_bands,
)


class BrainNetworkCoreTests(unittest.TestCase):
    def test_frozen_roi_map_covers_52_non_midline_channels(self) -> None:
        channels = mapped_channels(PAPER_ROI_CHANNELS)
        self.assertEqual(len(channels), 52)
        self.assertEqual(len(set(channels)), 52)
        self.assertNotIn("Cz", channels)

    def test_unique_channel_edges_cover_within_and_interregional_families(self) -> None:
        channels = mapped_channels(PAPER_ROI_CHANNELS)
        edges = build_roi_channel_edges(channels, PAPER_ROI_CHANNELS)
        scopes = [edge.edge_scope for edge in edges]
        self.assertEqual(len(edges), 1326)
        self.assertEqual(scopes.count("within_roi"), 138)
        self.assertEqual(scopes.count("interregional"), 1188)
        self.assertEqual(
            len({(edge.source_index, edge.target_index) for edge in edges}),
            len(edges),
        )

    def test_roi_aggregation_produces_55_pairs_per_band(self) -> None:
        channels = mapped_channels(PAPER_ROI_CHANNELS)
        edges = build_roi_channel_edges(channels, PAPER_ROI_CHANNELS)
        values = np.ones((len(edges), len(PAPER_BANDS_HZ)))
        rows = aggregate_roi_connectivity(values, edges, band_names=list(PAPER_BANDS_HZ))
        self.assertEqual(len(rows), 55 * 5)
        self.assertTrue(all(row["absolute_imaginary_coherence"] == 1.0 for row in rows))

    def test_bands_may_share_boundaries_but_not_overlap(self) -> None:
        validate_bands(PAPER_BANDS_HZ)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            validate_bands({"one": [2, 5], "two": [4, 8]})


if __name__ == "__main__":
    unittest.main()

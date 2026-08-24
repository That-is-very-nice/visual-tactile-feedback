from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from visual_tactile_force.brain_network import RoiChannelEdge
from visual_tactile_force.brain_network_pipeline import compute_multitaper_absolute_imcoh


class _Connectivity:
    def get_data(self) -> np.ndarray:
        return np.array([[-0.1, 0.2], [0.3, -0.4]])


class BrainNetworkPipelineTests(unittest.TestCase):
    def test_imcoh_is_absolute_and_band_averaged(self) -> None:
        captured = {}

        def fake_connectivity(data: np.ndarray, **kwargs: object) -> _Connectivity:
            captured.update(kwargs)
            return _Connectivity()

        fake_module = types.SimpleNamespace(spectral_connectivity_epochs=fake_connectivity)
        edges = [
            RoiChannelEdge("A", "B", "A1", "B1", 0, 1, "interregional"),
            RoiChannelEdge("A", "B", "A2", "B1", 2, 1, "interregional"),
        ]
        with patch.dict(sys.modules, {"mne_connectivity": fake_module}):
            values = compute_multitaper_absolute_imcoh(
                np.zeros((4, 3, 100)),
                sampling_rate_hz=100.0,
                edges=edges,
                bands_hz={"theta": [4, 8], "alpha": [8, 13]},
                multitaper_bandwidth_hz=2.0,
                multitaper_adaptive=True,
            )
        np.testing.assert_array_equal(values, np.array([[0.1, 0.2], [0.3, 0.4]]))
        self.assertEqual(captured["method"], "imcoh")
        self.assertTrue(captured["faverage"])

    def test_phase_lagged_signals_have_larger_absolute_imcoh_than_noise(self) -> None:
        try:
            import mne_connectivity  # noqa: F401
        except ImportError:
            self.skipTest("optional mne-connectivity dependency is not installed")
        random = np.random.RandomState(7)
        sampling_rate = 100.0
        time = np.arange(1000) / sampling_rate
        segments = np.empty((20, 3, time.size), dtype=float)
        for epoch in range(segments.shape[0]):
            phase = random.uniform(0, 2 * np.pi)
            segments[epoch, 0] = np.sin(2 * np.pi * 10 * time + phase) + random.normal(
                scale=0.2, size=time.size
            )
            segments[epoch, 1] = np.cos(2 * np.pi * 10 * time + phase) + random.normal(
                scale=0.2, size=time.size
            )
            segments[epoch, 2] = random.normal(size=time.size)
        edges = [
            RoiChannelEdge("A", "B", "A1", "B1", 0, 1, "interregional"),
            RoiChannelEdge("A", "C", "A1", "C1", 0, 2, "interregional"),
        ]
        values = compute_multitaper_absolute_imcoh(
            segments,
            sampling_rate_hz=sampling_rate,
            edges=edges,
            bands_hz={"alpha": [8, 13]},
            multitaper_bandwidth_hz=2.0,
            multitaper_adaptive=True,
        )
        self.assertGreater(values[0, 0], values[1, 0] + 0.3)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from visual_tactile_force.cmc_pipeline import compute_multitaper_cmc


@unittest.skipUnless(
    importlib.util.find_spec("mne_connectivity"),
    "optional mne-connectivity dependency is not installed",
)
class NeuroCmcIntegrationTests(unittest.TestCase):
    def test_shared_ten_hz_signal_has_higher_coherence_than_noise(self) -> None:
        rng = np.random.default_rng(20260821)
        sfreq = 100.0
        time = np.arange(500) / sfreq
        eeg = np.empty((20, 2, time.size))
        emg = np.empty((20, 1, time.size))
        for segment in range(20):
            phase = rng.uniform(0, 2 * np.pi)
            shared = np.sin(2 * np.pi * 10.0 * time + phase)
            emg[segment, 0] = shared + 0.25 * rng.normal(size=time.size)
            eeg[segment, 0] = shared + 0.25 * rng.normal(size=time.size)
            eeg[segment, 1] = rng.normal(size=time.size)

        spectra, frequencies = compute_multitaper_cmc(
            eeg,
            emg,
            sampling_rate_hz=sfreq,
            fmin_hz=5.0,
            fmax_hz=20.0,
            multitaper_bandwidth_hz=2.0,
            multitaper_adaptive=True,
        )
        ten_hz = int(np.argmin(np.abs(frequencies - 10.0)))
        self.assertGreater(spectra[0, ten_hz], 0.8)
        self.assertLess(spectra[1, ten_hz], 0.3)


if __name__ == "__main__":
    unittest.main()

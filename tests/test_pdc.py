from __future__ import annotations

import unittest

import numpy as np

from visual_tactile_force.pdc import (
    fit_bivariate_pdc,
    monte_carlo_pdc_thresholds,
    pdc_frequency_axis,
    standardize_bivariate_epochs,
    summarize_roi_pdc,
)


class PdcCoreTests(unittest.TestCase):
    def test_frequency_axis_includes_zero_and_nyquist(self) -> None:
        frequencies = pdc_frequency_axis(sampling_rate_hz=160.0, n_frequency_bins=801)
        self.assertEqual(frequencies.shape, (801,))
        self.assertAlmostEqual(frequencies[0], 0.0)
        self.assertAlmostEqual(frequencies[-1], 80.0)
        self.assertAlmostEqual(frequencies[1] - frequencies[0], 0.1)

    def test_standardization_is_per_epoch_and_channel(self) -> None:
        values = np.arange(4 * 2 * 20, dtype=float).reshape(4, 2, 20)
        standardized = standardize_bivariate_epochs(values)
        np.testing.assert_allclose(standardized.mean(axis=-1), 0.0, atol=1e-14)
        np.testing.assert_allclose(standardized.std(axis=-1), 1.0, atol=1e-14)

    def test_summary_maps_legacy_metrics_to_descriptive_names(self) -> None:
        frequencies = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
        spectra = np.array([[0.1, 0.3, 0.4, 0.1, 0.2, 0.5], [0.2] * 6])
        threshold = np.full(6, 0.2)
        result = summarize_roi_pdc(
            spectra,
            frequencies,
            threshold,
            channel_names=["C3", "Cz"],
            roi_channels=["C3", "Cz"],
            bands_hz={"alpha": [8.0, 13.0]},
        )["alpha"]
        self.assertAlmostEqual(result["max_normalized_suprathreshold_excess"], 0.12)
        self.assertAlmostEqual(result["max_mean_suprathreshold_excess"], 0.2)
        self.assertAlmostEqual(result["max_band_mean"], 0.26666666666666666)

    def test_synthetic_eeg_to_emg_process_has_larger_descending_pdc(self) -> None:
        random = np.random.RandomState(4)
        data = np.zeros((20, 2, 600), dtype=float)
        for epoch in range(data.shape[0]):
            for sample in range(2, data.shape[-1]):
                data[epoch, 0, sample] = 0.75 * data[epoch, 0, sample - 1] + random.normal()
                data[epoch, 1, sample] = (
                    0.55 * data[epoch, 1, sample - 1]
                    + 0.8 * data[epoch, 0, sample - 1]
                    + random.normal(scale=0.5)
                )
        spectra = fit_bivariate_pdc(data, model_order=2, n_frequency_bins=101)
        self.assertGreater(np.mean(spectra["descending"]), np.mean(spectra["ascending"]) * 2)

    def test_small_monte_carlo_run_is_reproducible_and_frequency_resolved(self) -> None:
        settings = dict(
            epoch_count=4,
            samples_per_epoch=80,
            model_order=2,
            n_frequency_bins=21,
            iterations=5,
            percentile=95.0,
            random_seed=0,
        )
        first = monte_carlo_pdc_thresholds(**settings)
        second = monte_carlo_pdc_thresholds(**settings)
        for direction in ("descending", "ascending"):
            self.assertEqual(first[direction].shape, (21,))
            np.testing.assert_array_equal(first[direction], second[direction])
            self.assertGreater(np.std(first[direction]), 0.0)


if __name__ == "__main__":
    unittest.main()

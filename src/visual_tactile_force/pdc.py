"""Pure numerical building blocks for bidirectional EEG–EMG PDC."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import linalg
from scipy.fftpack import fft


PAPER_PDC_METRIC = "max_mean_suprathreshold_excess"
PDC_METRIC_DEFINITIONS = {
    "max_normalized_suprathreshold_excess": (
        "ROI maximum of channel-wise integrated positive PDC excess, "
        "divided by band width"
    ),
    "max_mean_suprathreshold_excess": (
        "ROI maximum of channel-wise mean positive PDC excess "
        "above the null threshold"
    ),
}
PAPER_PDC_METRIC_DEFINITION = PDC_METRIC_DEFINITIONS[PAPER_PDC_METRIC]
PDC_DIRECTIONS = {
    "descending": (1, 0),  # EEG (source 0) -> EMG (target 1)
    "ascending": (0, 1),  # EMG (source 1) -> EEG (target 0)
}
PDC_CANDIDATE_METRICS = (
    "mean_normalized_suprathreshold_excess",
    "mean_mean_suprathreshold_excess",
    "mean_band_mean",
    "max_normalized_suprathreshold_excess",
    "max_mean_suprathreshold_excess",
    "max_band_mean",
)


def pdc_frequency_axis(*, sampling_rate_hz: float, n_frequency_bins: int) -> np.ndarray:
    """Return the SCoT one-sided frequency axis, including zero and Nyquist."""

    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if n_frequency_bins < 2:
        raise ValueError("n_frequency_bins must be at least two")
    return np.linspace(0.0, sampling_rate_hz / 2.0, n_frequency_bins)


def standardize_bivariate_epochs(data: np.ndarray) -> np.ndarray:
    """Z-standardize every epoch/channel time series independently."""

    values = np.asarray(data, dtype=float)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("data must have shape (epochs, 2, samples)")
    if values.shape[0] < 2:
        raise ValueError("at least two epochs are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("PDC input must contain only finite values")
    means = np.mean(values, axis=-1, keepdims=True)
    standard_deviations = np.std(values, axis=-1, keepdims=True)
    if np.any(standard_deviations == 0):
        raise ValueError("PDC input contains a constant epoch/channel")
    return (values - means) / standard_deviations


def fit_bivariate_pdc(
    data: np.ndarray,
    *,
    model_order: int,
    n_frequency_bins: int,
    n_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """Fit an OLS bivariate VAR and return Baccalá PDC in both directions.

    The coefficient layout and Fourier transform intentionally match SCoT's
    ``VAR`` and ``Connectivity.PDC`` implementations. Keeping these few lines
    here avoids a runtime dependency on the 2016 PyPI release of SCoT, which
    calls SciPy APIs removed from modern versions.
    """

    if model_order < 1:
        raise ValueError("model_order must be positive")
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    standardized = standardize_bivariate_epochs(data)
    if standardized.shape[-1] <= model_order:
        raise ValueError("model_order must be smaller than samples per epoch")
    epoch_count, channel_count, sample_count = standardized.shape
    relation_count = (sample_count - model_order) * epoch_count
    predictors = np.zeros((relation_count, channel_count * model_order), dtype=float)
    responses = np.zeros((relation_count, channel_count), dtype=float)
    for channel in range(channel_count):
        for lag in range(1, model_order + 1):
            predictors[:, channel * model_order + lag - 1] = np.reshape(
                standardized[:, channel, model_order - lag : -lag].T,
                relation_count,
            )
        responses[:, channel] = np.reshape(
            standardized[:, channel, model_order:].T, relation_count
        )
    coefficients = linalg.lstsq(predictors, responses)[0].T
    coefficient_lags = np.reshape(
        coefficients, (channel_count, channel_count, model_order), order="C"
    )
    spectral_coefficients = fft(
        np.dstack([np.eye(channel_count), -coefficient_lags]),
        int(n_frequency_bins) * 2 - 1,
    )[:, :, : int(n_frequency_bins)]
    denominator = np.sqrt(
        np.sum(
            spectral_coefficients.conj() * spectral_coefficients,
            axis=0,
            keepdims=True,
        )
    )
    spectra = np.asarray(np.abs(spectral_coefficients / denominator), dtype=float)
    expected_shape = (2, 2, int(n_frequency_bins))
    if spectra.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected SCoT PDC shape {spectra.shape}; expected {expected_shape}"
        )
    return {
        name: spectra[target, source].copy()
        for name, (target, source) in PDC_DIRECTIONS.items()
    }


def monte_carlo_pdc_thresholds(
    *,
    epoch_count: int,
    samples_per_epoch: int,
    model_order: int,
    n_frequency_bins: int,
    iterations: int,
    percentile: float,
    random_seed: int,
    n_jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Estimate frequency-resolved null thresholds from independent Gaussian pairs."""

    if epoch_count < 2 or samples_per_epoch < 2:
        raise ValueError("Monte Carlo shape must contain at least two epochs and samples")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0 < percentile < 100:
        raise ValueError("percentile must lie between zero and 100")

    # RandomState intentionally freezes NumPy's legacy MT19937 stream. The old
    # notebook called np.random.seed(0); using the explicit object avoids global state.
    random = np.random.RandomState(int(random_seed))
    null_spectra = {
        direction: np.empty((iterations, n_frequency_bins), dtype=float)
        for direction in PDC_DIRECTIONS
    }
    for iteration in range(iterations):
        noise = random.normal(size=(epoch_count, 2, samples_per_epoch))
        spectra = fit_bivariate_pdc(
            noise,
            model_order=model_order,
            n_frequency_bins=n_frequency_bins,
            n_jobs=n_jobs,
        )
        for direction in PDC_DIRECTIONS:
            null_spectra[direction][iteration] = spectra[direction]
        if progress is not None:
            progress(iteration + 1, iterations)
    return {
        direction: np.percentile(values, percentile, axis=0)
        for direction, values in null_spectra.items()
    }


def summarize_roi_pdc(
    spectra: np.ndarray,
    frequencies_hz: np.ndarray,
    threshold: np.ndarray,
    *,
    channel_names: Sequence[str],
    roi_channels: Sequence[str],
    bands_hz: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float]]:
    """Calculate six explicitly named replacements for legacy PDC metrics 1–6."""

    values = np.asarray(spectra, dtype=float)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    threshold_values = np.asarray(threshold, dtype=float)
    if values.ndim != 2:
        raise ValueError("spectra must have shape (channels, frequencies)")
    if values.shape != (len(channel_names), frequencies.size):
        raise ValueError("spectra shape does not match channel/frequency metadata")
    if threshold_values.shape != frequencies.shape:
        raise ValueError("threshold must have one value per frequency")
    if not all(np.all(np.isfinite(item)) for item in (values, frequencies, threshold_values)):
        raise ValueError("spectra, frequencies, and threshold must be finite")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies must be strictly increasing")

    channel_index = {name: index for index, name in enumerate(channel_names)}
    missing = [name for name in roi_channels if name not in channel_index]
    if missing:
        raise ValueError(f"ROI channels missing from data: {missing}")
    roi = values[[channel_index[name] for name in roi_channels]]

    summaries: dict[str, dict[str, float]] = {}
    for band, bounds in bands_hz.items():
        if len(bounds) != 2:
            raise ValueError(f"Band {band!r} must contain [low, high]")
        low, high = float(bounds[0]), float(bounds[1])
        if low >= high:
            raise ValueError(f"Band {band!r} requires low < high")
        mask = (frequencies >= low) & (frequencies <= high)
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"Band {band!r} must contain at least two frequency bins")
        band_values = roi[:, mask]
        excess = np.maximum(band_values - threshold_values[mask], 0.0)
        spacing = float(np.median(np.diff(frequencies[mask])))
        normalized_areas = np.sum(excess, axis=1) * spacing / (high - low)
        positive_means = np.zeros(excess.shape[0], dtype=float)
        for channel, channel_excess in enumerate(excess):
            positive = channel_excess[channel_excess > 0]
            positive_means[channel] = float(np.mean(positive)) if positive.size else 0.0
        band_means = np.mean(band_values, axis=1)
        summaries[band] = {
            "mean_normalized_suprathreshold_excess": float(np.mean(normalized_areas)),
            "mean_mean_suprathreshold_excess": float(np.mean(positive_means)),
            "mean_band_mean": float(np.mean(band_means)),
            "max_normalized_suprathreshold_excess": float(np.max(normalized_areas)),
            "max_mean_suprathreshold_excess": float(np.max(positive_means)),
            "max_band_mean": float(np.max(band_means)),
        }
    return summaries

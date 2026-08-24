"""MNE/SCoT-backed corrected PDC computation for one EEG–EMG input."""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import resample_poly

from .cmc import segment_trials, validate_aligned_segments
from .neuro_registry import NeuroInput
from .pdc import (
    PAPER_PDC_METRIC,
    PDC_DIRECTIONS,
    fit_bivariate_pdc,
    pdc_frequency_axis,
    summarize_roi_pdc,
)


def resample_segments(
    data: np.ndarray,
    *,
    source_rate_hz: float,
    target_rate_hz: float,
) -> np.ndarray:
    """Anti-alias and resample segmented signals along their time axis."""

    values = np.asarray(data, dtype=float)
    if values.ndim != 3:
        raise ValueError("data must have shape (segments, channels, samples)")
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("sampling rates must be positive")
    ratio = Fraction(str(target_rate_hz)) / Fraction(str(source_rate_hz))
    return resample_poly(values, up=ratio.numerator, down=ratio.denominator, axis=-1)


def compute_pdc_trace(
    item: NeuroInput,
    *,
    trial_indices: Sequence[int],
    tmin_s: float,
    tmax_s: float,
    segment_duration_s: float,
    apply_csd: bool,
    csd_stiffness: float,
    csd_n_legendre_terms: int,
    csd_lambda2: float,
    target_sampling_rate_hz: float,
    model_order: int,
    n_frequency_bins: int,
    spectral_range_hz: Sequence[float],
    roi_channels: Sequence[str],
    bands_hz: Mapping[str, Sequence[float]],
    thresholds: Mapping[str, np.ndarray],
    summary_metric: str = PAPER_PDC_METRIC,
    include_candidate_metrics: bool = False,
    n_jobs: int = 1,
) -> dict[str, object]:
    """Run a traceable, frequency-correct PDC calculation for one dataset."""

    try:
        import mne
    except ImportError as error:  # pragma: no cover - optional environment
        raise ImportError("PDC computation requires the 'pdc' optional dependencies") from error

    eeg_epochs = mne.io.read_epochs_eeglab(item.eeg_set, verbose="ERROR")
    emg_epochs = mne.io.read_epochs_eeglab(item.emg_set, verbose="ERROR")
    eeg_rate = float(eeg_epochs.info["sfreq"])
    emg_rate = float(emg_epochs.info["sfreq"])
    if not np.isclose(eeg_rate, emg_rate):
        raise ValueError(f"EEG/EMG sampling rates differ: {eeg_rate} versus {emg_rate}")
    if apply_csd:
        eeg_epochs = mne.preprocessing.compute_current_source_density(
            eeg_epochs,
            stiffness=float(csd_stiffness),
            n_legendre_terms=int(csd_n_legendre_terms),
            lambda2=float(csd_lambda2),
            copy=True,
        )

    eeg_segments, eeg_windows = segment_trials(
        eeg_epochs.get_data(copy=False),
        sampling_rate_hz=eeg_rate,
        trial_indices=trial_indices,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        segment_duration_s=segment_duration_s,
    )
    emg_segments, emg_windows = segment_trials(
        emg_epochs.get_data(copy=False),
        sampling_rate_hz=emg_rate,
        trial_indices=trial_indices,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        segment_duration_s=segment_duration_s,
    )
    if eeg_windows != emg_windows:
        raise ValueError("EEG and EMG segment provenance differs")
    validate_aligned_segments(eeg_segments, emg_segments)
    if emg_segments.shape[1] != 1:
        raise ValueError("PDC requires exactly one EMG channel")
    missing_roi = [channel for channel in roi_channels if channel not in eeg_epochs.ch_names]
    if missing_roi:
        raise ValueError(f"ROI channels missing from EEG data: {missing_roi}")
    roi_indices = [eeg_epochs.ch_names.index(channel) for channel in roi_channels]
    roi_eeg = resample_segments(
        eeg_segments[:, roi_indices],
        source_rate_hz=eeg_rate,
        target_rate_hz=target_sampling_rate_hz,
    )
    emg = resample_segments(
        emg_segments,
        source_rate_hz=emg_rate,
        target_rate_hz=target_sampling_rate_hz,
    )
    validate_aligned_segments(roi_eeg, emg)

    direction_spectra = {
        direction: np.empty((len(roi_channels), n_frequency_bins), dtype=float)
        for direction in PDC_DIRECTIONS
    }
    for channel_index in range(len(roi_channels)):
        pair = np.concatenate([roi_eeg[:, channel_index : channel_index + 1], emg], axis=1)
        channel_spectra = fit_bivariate_pdc(
            pair,
            model_order=model_order,
            n_frequency_bins=n_frequency_bins,
            n_jobs=n_jobs,
        )
        for direction in PDC_DIRECTIONS:
            direction_spectra[direction][channel_index] = channel_spectra[direction]

    frequencies = pdc_frequency_axis(
        sampling_rate_hz=target_sampling_rate_hz,
        n_frequency_bins=n_frequency_bins,
    )
    low, high = float(spectral_range_hz[0]), float(spectral_range_hz[1])
    spectral_mask = (frequencies >= low) & (frequencies <= high)
    if not np.any(spectral_mask):
        raise ValueError("spectral_range_hz contains no PDC frequency bins")

    candidate_metrics: dict[str, dict[str, dict[str, float]]] = {}
    stable_values: dict[str, dict[str, float]] = {}
    for direction in PDC_DIRECTIONS:
        if direction not in thresholds:
            raise ValueError(f"Missing Monte Carlo threshold for {direction}")
        threshold = np.asarray(thresholds[direction], dtype=float)
        if threshold.shape != frequencies.shape:
            raise ValueError(f"Threshold shape mismatch for {direction}")
        summaries = summarize_roi_pdc(
            direction_spectra[direction][:, spectral_mask],
            frequencies[spectral_mask],
            threshold[spectral_mask],
            channel_names=roi_channels,
            roi_channels=roi_channels,
            bands_hz=bands_hz,
        )
        if any(summary_metric not in metrics for metrics in summaries.values()):
            raise ValueError(f"Unknown PDC summary metric: {summary_metric!r}")
        candidate_metrics[direction] = summaries
        stable_values[direction] = {
            band: float(metrics[summary_metric]) for band, metrics in summaries.items()
        }

    result: dict[str, object] = {
        "subject": item.subject,
        "condition": item.condition,
        "event_code": item.event_code,
        "segment_count": int(roi_eeg.shape[0]),
        "samples_per_segment": int(roi_eeg.shape[-1]),
        "input_sampling_rate_hz": eeg_rate,
        "sampling_rate_hz": float(target_sampling_rate_hz),
        "model_order": int(model_order),
        "frequency_bin_count": int(n_frequency_bins),
        "frequency_min_hz": float(frequencies[spectral_mask].min()),
        "frequency_max_hz": float(frequencies[spectral_mask].max()),
        "summary_metric": summary_metric,
        "pdc_by_direction_band": stable_values,
        "software": {
            "mne": mne.__version__,
            "pdc_implementation": "internal SCoT-compatible Baccala 2001 formula",
        },
    }
    if include_candidate_metrics:
        result["roi_candidate_metrics"] = candidate_metrics
    return result

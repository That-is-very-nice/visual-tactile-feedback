"""Optional MNE-backed CMC computation built on the paper contract."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .cmc import (
    PAPER_CMC_METRIC,
    coherence_confidence_limit,
    segment_trials,
    summarize_roi_cmc,
    validate_aligned_segments,
)
from .neuro_registry import NeuroInput


def compute_multitaper_cmc(
    eeg_segments: np.ndarray,
    emg_segments: np.ndarray,
    *,
    sampling_rate_hz: float,
    fmin_hz: float,
    fmax_hz: float,
    multitaper_bandwidth_hz: float,
    multitaper_adaptive: bool,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute EMG-to-each-EEG magnitude-squared coherence spectra."""

    try:
        from mne_connectivity import spectral_connectivity_epochs
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "CMC computation requires the 'neuro' optional dependencies."
        ) from error

    eeg = np.asarray(eeg_segments, dtype=float)
    emg = np.asarray(emg_segments, dtype=float)
    validate_aligned_segments(eeg, emg)
    if emg.shape[1] != 1:
        raise ValueError("CMC currently requires exactly one EMG channel")

    combined = np.concatenate([eeg, emg], axis=1)
    emg_index = eeg.shape[1]
    indices = (
        np.full(eeg.shape[1], emg_index, dtype=int),
        np.arange(eeg.shape[1], dtype=int),
    )
    connectivity = spectral_connectivity_epochs(
        combined,
        method="coh",
        indices=indices,
        sfreq=float(sampling_rate_hz),
        mode="multitaper",
        fmin=float(fmin_hz),
        fmax=float(fmax_hz),
        faverage=False,
        mt_bandwidth=float(multitaper_bandwidth_hz),
        mt_adaptive=bool(multitaper_adaptive),
        n_jobs=int(n_jobs),
        verbose="ERROR",
    )
    spectra = np.asarray(connectivity.get_data(), dtype=float)
    frequencies = np.asarray(connectivity.freqs, dtype=float)
    if spectra.shape != (eeg.shape[1], frequencies.size):
        raise RuntimeError(
            "Unexpected connectivity shape: "
            f"{spectra.shape}, expected {(eeg.shape[1], frequencies.size)}"
        )
    return spectra, frequencies


def compute_cmc_trace(
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
    spectral_range_hz: Sequence[float],
    multitaper_bandwidth_hz: float,
    multitaper_adaptive: bool,
    roi_channels: Sequence[str],
    bands_hz: Mapping[str, Sequence[float]],
    confidence_alpha: float,
    summary_metric: str = PAPER_CMC_METRIC,
    include_candidate_metrics: bool = False,
    n_jobs: int = 1,
) -> dict[str, object]:
    """Run a traceable CMC calculation for one subject-condition input."""

    try:
        import mne
        import mne_connectivity
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "CMC computation requires the 'neuro' optional dependencies."
        ) from error

    eeg_epochs = mne.io.read_epochs_eeglab(item.eeg_set, verbose="ERROR")
    emg_epochs = mne.io.read_epochs_eeglab(item.emg_set, verbose="ERROR")
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
        sampling_rate_hz=float(eeg_epochs.info["sfreq"]),
        trial_indices=trial_indices,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        segment_duration_s=segment_duration_s,
    )
    emg_segments, emg_windows = segment_trials(
        emg_epochs.get_data(copy=False),
        sampling_rate_hz=float(emg_epochs.info["sfreq"]),
        trial_indices=trial_indices,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        segment_duration_s=segment_duration_s,
    )
    if eeg_windows != emg_windows:
        raise ValueError("EEG and EMG segment provenance differs")
    missing_roi = [name for name in roi_channels if name not in eeg_epochs.ch_names]
    if missing_roi:
        raise ValueError(f"ROI channels missing from EEG data: {missing_roi}")
    roi_indices = [eeg_epochs.ch_names.index(name) for name in roi_channels]
    roi_eeg_segments = eeg_segments[:, roi_indices, :]
    spectra, frequencies = compute_multitaper_cmc(
        roi_eeg_segments,
        emg_segments,
        sampling_rate_hz=float(eeg_epochs.info["sfreq"]),
        fmin_hz=float(spectral_range_hz[0]),
        fmax_hz=float(spectral_range_hz[1]),
        multitaper_bandwidth_hz=multitaper_bandwidth_hz,
        multitaper_adaptive=multitaper_adaptive,
        n_jobs=n_jobs,
    )
    confidence_limit = coherence_confidence_limit(
        alpha=confidence_alpha,
        independent_segment_count=eeg_segments.shape[0],
    )
    summaries = summarize_roi_cmc(
        spectra,
        frequencies,
        channel_names=roi_channels,
        roi_channels=roi_channels,
        bands_hz=bands_hz,
        confidence_limit=confidence_limit,
    )
    missing_metric_bands = [
        band for band, metrics in summaries.items() if summary_metric not in metrics
    ]
    if missing_metric_bands:
        raise ValueError(
            f"Unknown CMC summary metric {summary_metric!r} for bands "
            f"{missing_metric_bands}"
        )
    result: dict[str, object] = {
        "subject": item.subject,
        "condition": item.condition,
        "event_code": item.event_code,
        "segment_count": int(eeg_segments.shape[0]),
        "samples_per_segment": int(eeg_segments.shape[-1]),
        "sampling_rate_hz": float(eeg_epochs.info["sfreq"]),
        "confidence_limit": confidence_limit,
        "frequency_bin_count": int(frequencies.size),
        "frequency_min_hz": float(frequencies.min()),
        "frequency_max_hz": float(frequencies.max()),
        "summary_metric": summary_metric,
        "cmc_by_band": {
            band: float(metrics[summary_metric])
            for band, metrics in summaries.items()
        },
        "software": {
            "mne": mne.__version__,
            "mne_connectivity": mne_connectivity.__version__,
        },
    }
    if include_candidate_metrics:
        result["roi_candidate_metrics"] = summaries
    return result

"""MNE-backed sensor-space absolute imaginary-coherence pipeline."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .brain_network import (
    PAPER_NETWORK_METRIC,
    aggregate_roi_connectivity,
    build_roi_channel_edges,
    validate_bands,
)
from .cmc import segment_trials
from .neuro_registry import NeuroInput


def compute_multitaper_absolute_imcoh(
    eeg_segments: np.ndarray,
    *,
    sampling_rate_hz: float,
    edges: Sequence[object],
    bands_hz: Mapping[str, Sequence[float]],
    multitaper_bandwidth_hz: float,
    multitaper_adaptive: bool,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute one band-averaged |ImCoh| value for every channel edge."""

    try:
        from mne_connectivity import spectral_connectivity_epochs
    except ImportError as error:  # pragma: no cover - optional environment
        raise ImportError(
            "Brain-network computation requires the 'network' optional dependencies"
        ) from error

    values = np.asarray(eeg_segments, dtype=float)
    if values.ndim != 3:
        raise ValueError("eeg_segments must have shape (segments, channels, samples)")
    if not edges:
        raise ValueError("At least one channel edge is required")
    validate_bands(bands_hz)
    source_indices = np.array([int(getattr(edge, "source_index")) for edge in edges])
    target_indices = np.array([int(getattr(edge, "target_index")) for edge in edges])
    lows = tuple(float(bounds[0]) for bounds in bands_hz.values())
    highs = tuple(float(bounds[1]) for bounds in bands_hz.values())
    connectivity = spectral_connectivity_epochs(
        values,
        method="imcoh",
        indices=(source_indices, target_indices),
        sfreq=float(sampling_rate_hz),
        mode="multitaper",
        fmin=lows,
        fmax=highs,
        faverage=True,
        mt_bandwidth=float(multitaper_bandwidth_hz),
        mt_adaptive=bool(multitaper_adaptive),
        n_jobs=int(n_jobs),
        verbose="ERROR",
    )
    spectra = np.abs(np.asarray(connectivity.get_data(), dtype=float))
    expected = (len(edges), len(bands_hz))
    if spectra.shape != expected:
        raise RuntimeError(f"Unexpected ImCoh shape {spectra.shape}; expected {expected}")
    return spectra


def compute_brain_network_trace(
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
    roi_channels: Mapping[str, Sequence[str]],
    bands_hz: Mapping[str, Sequence[float]],
    multitaper_bandwidth_hz: float,
    multitaper_adaptive: bool,
    include_within_roi: bool = True,
    n_jobs: int = 1,
) -> dict[str, object]:
    """Run the frozen brain-network calculation for one subject-condition dataset."""

    try:
        import mne
        import mne_connectivity
    except ImportError as error:  # pragma: no cover - optional environment
        raise ImportError(
            "Brain-network computation requires the 'network' optional dependencies"
        ) from error

    eeg_epochs = mne.io.read_epochs_eeglab(item.eeg_set, verbose="ERROR")
    sampling_rate_hz = float(eeg_epochs.info["sfreq"])
    if apply_csd:
        eeg_epochs = mne.preprocessing.compute_current_source_density(
            eeg_epochs,
            stiffness=float(csd_stiffness),
            n_legendre_terms=int(csd_n_legendre_terms),
            lambda2=float(csd_lambda2),
            copy=True,
        )

    eeg_segments, windows = segment_trials(
        eeg_epochs.get_data(copy=False),
        sampling_rate_hz=sampling_rate_hz,
        trial_indices=trial_indices,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        segment_duration_s=segment_duration_s,
    )
    edges = build_roi_channel_edges(
        eeg_epochs.ch_names,
        roi_channels,
        include_within_roi=include_within_roi,
    )
    edge_values = compute_multitaper_absolute_imcoh(
        eeg_segments,
        sampling_rate_hz=sampling_rate_hz,
        edges=edges,
        bands_hz=bands_hz,
        multitaper_bandwidth_hz=multitaper_bandwidth_hz,
        multitaper_adaptive=multitaper_adaptive,
        n_jobs=n_jobs,
    )
    roi_rows = aggregate_roi_connectivity(
        edge_values,
        edges,
        band_names=list(bands_hz),
    )
    return {
        "subject": item.subject,
        "condition": item.condition,
        "event_code": item.event_code,
        "segment_count": int(eeg_segments.shape[0]),
        "samples_per_segment": int(eeg_segments.shape[-1]),
        "sampling_rate_hz": sampling_rate_hz,
        "window_count": len(windows),
        "channel_edge_count": len(edges),
        "summary_metric": PAPER_NETWORK_METRIC,
        "roi_connectivity": roi_rows,
        "software": {
            "mne": mne.__version__,
            "mne_connectivity": mne_connectivity.__version__,
        },
    }

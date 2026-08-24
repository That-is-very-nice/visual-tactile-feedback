"""Pure segmentation and CMC summary functions for the paper pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


PAPER_CMC_METRIC = "max_mean_suprathreshold_excess"
PAPER_CMC_METRIC_DEFINITION = (
    "ROI maximum of the channel-wise mean positive suprathreshold excess"
)


@dataclass(frozen=True)
class SegmentWindow:
    """Provenance for one output segment."""

    trial_index: int
    start_s: float
    stop_s: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def segment_trials(
    data: np.ndarray,
    *,
    sampling_rate_hz: float,
    trial_indices: Sequence[int],
    tmin_s: float,
    tmax_s: float,
    segment_duration_s: float,
) -> tuple[np.ndarray, list[SegmentWindow]]:
    """Select one-based trials, crop them, and make non-overlapping segments.

    ``data`` must be shaped ``(trials, channels, samples)`` or
    ``(trials, samples)``. The output always retains a channel dimension.
    """

    values = np.asarray(data)
    if values.ndim == 2:
        values = values[:, np.newaxis, :]
    if values.ndim != 3:
        raise ValueError("data must have shape (trials, channels, samples)")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if not 0 <= tmin_s < tmax_s:
        raise ValueError("require 0 <= tmin_s < tmax_s")
    if segment_duration_s <= 0:
        raise ValueError("segment_duration_s must be positive")

    def exact_samples(seconds: float, label: str) -> int:
        samples = seconds * sampling_rate_hz
        rounded = int(round(samples))
        if not np.isclose(samples, rounded, rtol=0.0, atol=1e-9):
            raise ValueError(f"{label} does not fall on a sample boundary")
        return rounded

    start_sample = exact_samples(tmin_s, "tmin_s")
    stop_sample = exact_samples(tmax_s, "tmax_s")
    segment_samples = exact_samples(segment_duration_s, "segment_duration_s")
    cropped_samples = stop_sample - start_sample
    if cropped_samples % segment_samples:
        raise ValueError("analysis window must contain a whole number of segments")
    if stop_sample > values.shape[-1]:
        raise ValueError(
            f"analysis stop requires {stop_sample} samples; data has {values.shape[-1]}"
        )

    segments: list[np.ndarray] = []
    windows: list[SegmentWindow] = []
    for trial_index_value in trial_indices:
        trial_index = int(trial_index_value)
        if trial_index < 1 or trial_index > values.shape[0]:
            raise IndexError(
                f"trial index {trial_index} is outside 1..{values.shape[0]}"
            )
        trial = values[trial_index - 1]
        for offset in range(0, cropped_samples, segment_samples):
            segment_start = start_sample + offset
            segments.append(trial[:, segment_start : segment_start + segment_samples])
            windows.append(
                SegmentWindow(
                    trial_index=trial_index,
                    start_s=tmin_s + offset / sampling_rate_hz,
                    stop_s=tmin_s + (offset + segment_samples) / sampling_rate_hz,
                )
            )

    return np.stack(segments, axis=0), windows


def validate_aligned_segments(eeg: np.ndarray, emg: np.ndarray) -> None:
    """Require equal segment and sample counts before spectral estimation."""

    eeg_values = np.asarray(eeg)
    emg_values = np.asarray(emg)
    if eeg_values.ndim != 3 or emg_values.ndim != 3:
        raise ValueError("EEG and EMG must both be 3-D segmented arrays")
    if eeg_values.shape[0] != emg_values.shape[0]:
        raise ValueError("EEG and EMG segment counts differ")
    if eeg_values.shape[-1] != emg_values.shape[-1]:
        raise ValueError("EEG and EMG sample counts differ")


def coherence_confidence_limit(
    *, alpha: float, independent_segment_count: int
) -> float:
    """Return ``1 - alpha**(1/(L-1))``, the legacy/paper CMC threshold."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one")
    if independent_segment_count <= 1:
        raise ValueError("independent_segment_count must be greater than one")
    return float(1.0 - alpha ** (1.0 / (independent_segment_count - 1)))


def summarize_roi_cmc(
    coherence: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    channel_names: Sequence[str],
    roi_channels: Sequence[str],
    bands_hz: Mapping[str, Sequence[float]],
    confidence_limit: float,
) -> dict[str, dict[str, float]]:
    """Compute explicitly named candidate ROI summaries for paper regression.

    The historical notebooks contain several anonymously numbered metrics.
    Returning descriptive alternatives prevents silently choosing the wrong one:

    - ``max_mean_suprathreshold_excess`` matches legacy ``cmc5``;
    - ``max_suprathreshold_excess`` is the largest point above threshold;
    - ``max_band_mean`` matches legacy ``cmc6``.

    Frequency boundaries are inclusive, matching the historical masks.
    """

    values = np.asarray(coherence, dtype=float)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if values.ndim != 2:
        raise ValueError("coherence must have shape (channels, frequencies)")
    if values.shape != (len(channel_names), frequencies.size):
        raise ValueError("coherence shape does not match channel/frequency metadata")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("coherence and frequencies must be finite")
    if not 0 <= confidence_limit <= 1:
        raise ValueError("confidence_limit must lie between zero and one")

    channel_index = {name: index for index, name in enumerate(channel_names)}
    missing = [name for name in roi_channels if name not in channel_index]
    if missing:
        raise ValueError(f"ROI channels missing from data: {missing}")
    roi = values[[channel_index[name] for name in roi_channels]]

    summaries: dict[str, dict[str, float]] = {}
    for band, bounds in bands_hz.items():
        if len(bounds) != 2:
            raise ValueError(f"Band {band!r} must contain [low, high]")
        low, high = (float(bounds[0]), float(bounds[1]))
        if low >= high:
            raise ValueError(f"Band {band!r} requires low < high")
        mask = (frequencies >= low) & (frequencies <= high)
        if not np.any(mask):
            raise ValueError(f"Band {band!r} contains no frequency bins")
        band_values = roi[:, mask]
        excess = np.maximum(band_values - confidence_limit, 0.0)
        frequency_spacing = float(np.median(np.diff(frequencies[mask])))
        normalized_areas = (
            np.sum(excess, axis=1) * frequency_spacing / (high - low)
        )
        positive_means = np.zeros(excess.shape[0], dtype=float)
        for channel_index_value, channel_excess in enumerate(excess):
            positive = channel_excess[channel_excess > 0]
            positive_means[channel_index_value] = (
                float(np.mean(positive)) if positive.size else 0.0
            )
        summaries[band] = {
            "mean_normalized_suprathreshold_area": float(
                np.mean(normalized_areas)
            ),
            "mean_mean_suprathreshold_excess": float(np.mean(positive_means)),
            "mean_band_mean": float(np.mean(np.mean(band_values, axis=1))),
            "max_normalized_suprathreshold_area": float(np.max(normalized_areas)),
            "max_mean_suprathreshold_excess": float(np.max(positive_means)),
            "max_suprathreshold_excess": float(np.max(excess)),
            "max_band_mean": float(np.max(np.mean(band_values, axis=1))),
        }
    return summaries

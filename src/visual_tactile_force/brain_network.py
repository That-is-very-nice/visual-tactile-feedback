"""Frozen sensor-space brain-network definitions used by the paper analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, product
from typing import Mapping, Sequence

import numpy as np


PAPER_ROI_CHANNELS: dict[str, tuple[str, ...]] = {
    "Left_Frontal": ("Fp1", "AF3", "AF7", "F1", "F3", "F5", "F7"),
    "Right_Frontal": ("Fp2", "AF4", "AF8", "F2", "F4", "F6", "F8"),
    "Left_Central": ("FC1", "FC3", "FC5", "C1", "C3", "C5", "CP1", "CP3", "CP5"),
    "Right_Central": ("FC2", "FC4", "FC6", "C2", "C4", "C6", "CP2", "CP4", "CP6"),
    "Left_Temporal": ("FT7", "T7", "TP7"),
    "Right_Temporal": ("FT8", "T8", "TP8"),
    "Left_Parietal": ("P1", "P3", "P5", "P7"),
    "Right_Parietal": ("P2", "P4", "P6", "P8"),
    "Left_Occipital": ("PO3", "PO7", "O1"),
    "Right_Occipital": ("PO4", "PO8", "O2"),
}

PAPER_BANDS_HZ: dict[str, tuple[float, float]] = {
    "delta": (2.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 58.0),
}

MIDLINE_CHANNELS = ("Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz")
PAPER_NETWORK_METRIC = "absolute_imaginary_coherence"
PAPER_NETWORK_METRIC_DEFINITION = (
    "mean across unique ROI electrode pairs of the absolute value of band-averaged "
    "imaginary coherence"
)


@dataclass(frozen=True)
class RoiChannelEdge:
    """One unique channel pair contributing to an ROI-level connection."""

    roi_source: str
    roi_target: str
    channel_source: str
    channel_target: str
    source_index: int
    target_index: int
    edge_scope: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_roi_map(roi_channels: Mapping[str, Sequence[str]]) -> None:
    """Reject empty ROIs and channels assigned to more than one ROI."""

    if len(roi_channels) != 10:
        raise ValueError(f"Brain-network analysis requires 10 ROIs; found {len(roi_channels)}")
    assigned: dict[str, str] = {}
    for roi, channels in roi_channels.items():
        if not channels:
            raise ValueError(f"ROI {roi!r} has no channels")
        for channel in channels:
            if channel in assigned:
                raise ValueError(
                    f"EEG channel {channel!r} is assigned to both {assigned[channel]!r} and {roi!r}"
                )
            assigned[channel] = roi


def mapped_channels(roi_channels: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Return ROI channels in stable ROI and within-ROI order."""

    validate_roi_map(roi_channels)
    return tuple(channel for channels in roi_channels.values() for channel in channels)


def build_roi_channel_edges(
    channel_names: Sequence[str],
    roi_channels: Mapping[str, Sequence[str]],
    *,
    include_within_roi: bool = True,
) -> list[RoiChannelEdge]:
    """Build unique unordered channel pairs for ROI-level connectivity.

    Absolute imaginary coherence is represented once for each unique channel pair.
    Within-ROI channel pairs exclude channel self-connections. Directed ROI rows
    required by the statistical correction are generated later at the ROI level.
    """

    validate_roi_map(roi_channels)
    if len(channel_names) != len(set(channel_names)):
        raise ValueError("EEG channel names must be unique")
    channel_index = {name: index for index, name in enumerate(channel_names)}
    missing = sorted(set(mapped_channels(roi_channels)) - set(channel_names))
    if missing:
        raise ValueError(f"ROI channels missing from EEG data: {missing}")

    rois = list(roi_channels)
    edges: list[RoiChannelEdge] = []
    for source_position, roi_source in enumerate(rois):
        target_start = source_position if include_within_roi else source_position + 1
        for roi_target in rois[target_start:]:
            if roi_source == roi_target:
                channel_pairs = combinations(roi_channels[roi_source], 2)
                edge_scope = "within_roi"
            else:
                channel_pairs = product(roi_channels[roi_source], roi_channels[roi_target])
                edge_scope = "interregional"
            for channel_source, channel_target in channel_pairs:
                edges.append(
                    RoiChannelEdge(
                        roi_source=roi_source,
                        roi_target=roi_target,
                        channel_source=channel_source,
                        channel_target=channel_target,
                        source_index=channel_index[channel_source],
                        target_index=channel_index[channel_target],
                        edge_scope=edge_scope,
                    )
                )
    return edges


def validate_bands(bands_hz: Mapping[str, Sequence[float]]) -> None:
    """Validate named closed frequency intervals while allowing shared boundaries."""

    if not bands_hz:
        raise ValueError("At least one frequency band is required")
    previous_high: float | None = None
    for band, bounds in bands_hz.items():
        if len(bounds) != 2:
            raise ValueError(f"Band {band!r} must contain [low, high]")
        low, high = (float(value) for value in bounds)
        if not 0 <= low < high:
            raise ValueError(f"Invalid frequency interval for {band!r}: {bounds}")
        if previous_high is not None and low < previous_high:
            raise ValueError("Frequency bands may touch but must not overlap")
        previous_high = high


def aggregate_roi_connectivity(
    edge_values: np.ndarray,
    edges: Sequence[RoiChannelEdge],
    *,
    band_names: Sequence[str],
) -> list[dict[str, object]]:
    """Average channel-edge |ImCoh| into one value per ROI pair and band."""

    values = np.asarray(edge_values, dtype=float)
    if values.shape != (len(edges), len(band_names)):
        raise ValueError(
            f"edge_values has shape {values.shape}; expected {(len(edges), len(band_names))}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Connectivity values must be finite")
    if np.any(values < 0):
        raise ValueError("Absolute imaginary coherence must be non-negative")

    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, edge in enumerate(edges):
        key = (edge.roi_source, edge.roi_target, edge.edge_scope)
        groups.setdefault(key, []).append(index)

    rows: list[dict[str, object]] = []
    for (roi_source, roi_target, edge_scope), indices in groups.items():
        roi_values = values[np.asarray(indices)]
        for band_index, band in enumerate(band_names):
            rows.append(
                {
                    "roi_source": roi_source,
                    "roi_target": roi_target,
                    "edge_scope": edge_scope,
                    "band": band,
                    "channel_pair_count": len(indices),
                    PAPER_NETWORK_METRIC: float(np.mean(roi_values[:, band_index])),
                }
            )
    return rows

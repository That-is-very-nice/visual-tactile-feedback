"""Force-sensor ingestion and behavioral metrics for the paper pipeline."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FORCE_COLUMNS = ("event", "time_ms", "target_force", "measured_force")


@dataclass(frozen=True)
class TrialWindow:
    trial_index: int
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class TrialMetrics:
    subject: str
    condition: str
    trial_index: int
    n_samples: int
    sampling_rate_hz: float
    normalization_scale: float
    mean_force: float
    force_standard_deviation: float
    force_cv: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def read_force_csv(path: str | Path) -> pd.DataFrame:
    """Read the legacy four-column force CSV and validate its numeric fields."""

    csv_path = Path(path)
    # Some legacy files contain an isolated row with trailing corrupt fields.
    # Restrict parsing to the four declared columns so one malformed row cannot
    # shift every subsequent column in the file.
    frame = pd.read_csv(
        csv_path,
        header=None,
        names=FORCE_COLUMNS,
        usecols=[0, 1, 2, 3],
        dtype={"event": str},
        low_memory=False,
    )
    if frame.empty:
        raise ValueError(f"Force CSV is empty: {csv_path}")

    for column in FORCE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid_time = frame["time_ms"].isna()
    if invalid_time.any():
        invalid_markers = frame.loc[invalid_time, "event"].isin(["start", "end"])
        if invalid_markers.any():
            bad_rows = frame.loc[invalid_time].index.tolist()[:5]
            raise ValueError(f"Invalid marker times in {csv_path}; rows={bad_rows}")
        bad_rows = frame.index[invalid_time].tolist()
        warnings.warn(
            f"Dropping {len(bad_rows)} non-marker row(s) with invalid times from {csv_path}; "
            f"first rows={bad_rows[:5]}",
            RuntimeWarning,
            stacklevel=2,
        )
        frame = frame.loc[~invalid_time].reset_index(drop=True)
    return frame


def trial_windows(frame: pd.DataFrame, *, start_offset_ms: float) -> list[TrialWindow]:
    """Pair start/end markers and apply the steady-state offset to each start."""

    starts = frame.loc[frame["event"] == "start", "time_ms"].to_numpy(dtype=float)
    ends = frame.loc[frame["event"] == "end", "time_ms"].to_numpy(dtype=float)
    if starts.size != ends.size:
        raise ValueError(
            f"Mismatched trial markers: {starts.size} starts and {ends.size} ends."
        )

    windows: list[TrialWindow] = []
    for index, (start, end) in enumerate(zip(starts, ends, strict=True), start=1):
        adjusted_start = float(start + start_offset_ms)
        if adjusted_start >= end:
            raise ValueError(
                f"Trial {index} has no samples after the start offset: "
                f"start={adjusted_start}, end={end}."
            )
        windows.append(TrialWindow(index, adjusted_start, float(end)))
    return windows


def slice_trial(frame: pd.DataFrame, window: TrialWindow) -> tuple[np.ndarray, ...]:
    """Return time, target, and measured force arrays inside one trial window."""

    mask = frame["time_ms"].between(window.start_ms, window.end_ms, inclusive="both")
    sliced = frame.loc[mask, list(FORCE_COLUMNS[1:])]
    if len(sliced) < 5:
        raise ValueError(f"Trial {window.trial_index} contains fewer than five samples.")
    if sliced[["target_force", "measured_force"]].isna().any().any():
        raise ValueError(f"Trial {window.trial_index} contains non-numeric force samples.")
    return tuple(sliced[column].to_numpy(dtype=float) for column in FORCE_COLUMNS[1:])


def sampling_rate_hz(time_ms: np.ndarray) -> float:
    """Estimate sampling rate from monotonically increasing millisecond timestamps."""

    time_array = np.asarray(time_ms, dtype=float)
    intervals_ms = np.diff(time_array)
    if intervals_ms.size == 0 or np.any(intervals_ms <= 0):
        raise ValueError("Time values must be strictly increasing.")
    median_interval_ms = float(np.median(intervals_ms))
    return 1000.0 / median_interval_ms


def lowpass_zero_phase(
    values: np.ndarray,
    sampling_rate: float,
    *,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    """Apply the legacy Butterworth/Gustafsson zero-phase low-pass filter."""

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    nyquist = sampling_rate / 2.0
    cutoff_used = min(float(cutoff_hz), 0.95 * nyquist)
    if cutoff_used <= 0:
        raise ValueError("Low-pass cutoff must be positive.")
    numerator, denominator = butter(order, cutoff_used / nyquist, btype="low")
    return filtfilt(numerator, denominator, np.asarray(values, dtype=float), method="gust")


def normalization_scale(target_force: np.ndarray, mode: str) -> float:
    """Return the force normalization scale used for one trial."""

    if mode == "none":
        return 1.0
    if mode != "per_target_median":
        raise ValueError(f"Unsupported normalization mode: {mode}")
    scale = float(np.nanmedian(np.asarray(target_force, dtype=float)))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Target-force normalization scale must be finite and non-zero.")
    return scale


def compute_trial_metrics(
    *,
    subject: str,
    condition: str,
    trial_index: int,
    time_ms: np.ndarray,
    target_force: np.ndarray,
    measured_force: np.ndarray,
    lowpass_hz: float | None = 5.0,
    filter_order: int = 4,
    normalization: str = "per_target_median",
    ddof: int = 1,
) -> TrialMetrics:
    """Compute normalized mean force and CV for one steady-state trial."""

    rate = sampling_rate_hz(time_ms)
    measured = np.asarray(measured_force, dtype=float)
    if lowpass_hz is not None:
        measured = lowpass_zero_phase(
            measured,
            rate,
            cutoff_hz=lowpass_hz,
            order=filter_order,
        )

    scale = normalization_scale(target_force, normalization)
    normalized = measured / scale
    mean_force = float(np.nanmean(normalized))
    standard_deviation = float(np.nanstd(normalized, ddof=ddof))
    if not np.isfinite(mean_force) or mean_force == 0:
        force_cv = np.nan
    else:
        force_cv = float(standard_deviation / mean_force)

    return TrialMetrics(
        subject=subject,
        condition=condition,
        trial_index=trial_index,
        n_samples=int(normalized.size),
        sampling_rate_hz=rate,
        normalization_scale=scale,
        mean_force=mean_force,
        force_standard_deviation=standard_deviation,
        force_cv=force_cv,
    )


def analyze_force_dataset(
    *,
    data_dir: str | Path,
    subjects: Sequence[str],
    conditions: Sequence[str],
    trial_indices: Iterable[int],
    start_offsets_ms: Mapping[str, float],
    file_template: str = "{subject}_{condition}.csv",
    subject_subdirectories: Mapping[str, str] | None = None,
    lowpass_hz: float | None = 5.0,
    filter_order: int = 4,
    normalization: str = "per_target_median",
    ddof: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the behavior pipeline and return trial- and subject-level tables."""

    root = Path(data_dir)
    requested_trials = tuple(int(index) for index in trial_indices)
    rows: list[dict[str, str | int | float]] = []

    for subject in subjects:
        for condition in conditions:
            subject_dir = (
                subject_subdirectories.get(subject, subject)
                if subject_subdirectories is not None
                else ""
            )
            path = root / file_template.format(
                subject=subject,
                condition=condition,
                subject_dir=subject_dir,
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing force CSV: {path}")
            frame = read_force_csv(path)
            windows_by_index = {
                window.trial_index: window
                for window in trial_windows(
                    frame,
                    start_offset_ms=float(start_offsets_ms.get(condition, 0.0)),
                )
            }
            for trial_index in requested_trials:
                if trial_index not in windows_by_index:
                    raise ValueError(
                        f"{path} has no trial {trial_index}; available={sorted(windows_by_index)}"
                    )
                try:
                    time_ms, target, measured = slice_trial(
                        frame,
                        windows_by_index[trial_index],
                    )
                except ValueError as error:
                    raise ValueError(f"{path}: {error}") from error
                metrics = compute_trial_metrics(
                    subject=subject,
                    condition=condition,
                    trial_index=trial_index,
                    time_ms=time_ms,
                    target_force=target,
                    measured_force=measured,
                    lowpass_hz=lowpass_hz,
                    filter_order=filter_order,
                    normalization=normalization,
                    ddof=ddof,
                )
                rows.append(metrics.to_dict())

    trial_table = pd.DataFrame(rows)
    summary_table = (
        trial_table.groupby(["subject", "condition"], as_index=False)
        .agg(
            mean_force=("mean_force", "mean"),
            force_cv=("force_cv", "mean"),
            force_standard_deviation=("force_standard_deviation", "mean"),
            n_trials=("trial_index", "count"),
        )
        .sort_values(["subject", "condition"], ignore_index=True)
    )
    return trial_table, summary_table

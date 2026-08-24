"""EEG/EMG input registry and metadata-only quality checks."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class NeuroInput:
    """Files for one subject and experimental condition."""

    subject: str
    condition: str
    event_code: int
    eeg_set: Path
    eeg_fdt: Path
    emg_set: Path
    emg_fdt: Path
    annotation: Path

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in record.items()
        }


def build_neuro_registry(
    *,
    set_dir: str | Path,
    annotation_dir: str | Path,
    subjects: Sequence[str],
    condition_event_codes: Mapping[str, int | Mapping[str, int]],
    epoch_bounds_label: str = "[0 60]",
) -> list[NeuroInput]:
    """Build the expected EEGLAB/annotation paths without reading signal data."""

    set_root = Path(set_dir)
    annotation_root = Path(annotation_dir)
    registry: list[NeuroInput] = []
    for subject in subjects:
        subject_mapping = condition_event_codes.get(subject)
        if isinstance(subject_mapping, Mapping):
            event_codes = subject_mapping
        else:
            event_codes = condition_event_codes
        for condition, event_code_value in event_codes.items():
            if isinstance(event_code_value, Mapping):
                raise ValueError(
                    f"No subject-specific event mapping configured for {subject!r}"
                )
            event_code = int(event_code_value)
            stem = f"{subject}{event_code}_{epoch_bounds_label}"
            registry.append(
                NeuroInput(
                    subject=subject,
                    condition=condition,
                    event_code=event_code,
                    eeg_set=set_root / f"{stem}_EEG.set",
                    eeg_fdt=set_root / f"{stem}_EEG.fdt",
                    emg_set=set_root / f"{stem}_EMG.set",
                    emg_fdt=set_root / f"{stem}_EMG.fdt",
                    annotation=(
                        annotation_root
                        / f"eeg_{subject}{event_code}_61_5_annotations.txt"
                    ),
                )
            )
    return registry


def _annotation_events(path: Path) -> list[tuple[float, str]]:
    events: list[tuple[float, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Onset", "Annotation"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("annotation file must contain Onset and Annotation columns")
        for row in reader:
            annotation = (row.get("Annotation") or "").strip()
            onset_text = (row.get("Onset") or "").strip()
            if annotation and onset_text:
                events.append((float(onset_text), annotation))
    return events


def _annotation_counts(events: Sequence[tuple[float, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, annotation in events:
        counts[annotation] = counts.get(annotation, 0) + 1
    return counts


def _pair_trial_events(
    events: Sequence[tuple[float, str]],
    *,
    start_annotation: str,
    stop_annotation: str,
) -> tuple[list[float], int, int]:
    """Pair each start with the next stop, ignoring stops before any start."""

    durations: list[float] = []
    open_start: float | None = None
    orphan_starts = 0
    orphan_stops = 0
    for onset, annotation in events:
        if annotation == start_annotation:
            if open_start is not None:
                orphan_starts += 1
            open_start = onset
        elif annotation == stop_annotation:
            if open_start is None:
                orphan_stops += 1
            else:
                durations.append(onset - open_start)
                open_start = None
    if open_start is not None:
        orphan_starts += 1
    return durations, orphan_starts, orphan_stops


def profile_neuro_inputs(
    registry: Sequence[NeuroInput],
    *,
    start_annotation: str = "DC trigger 13",
    stop_annotation: str = "DC trigger 14",
    expected_epoch_count: int = 5,
    expected_trial_duration_s: float = 60.0,
    trial_duration_tolerance_s: float = 0.1,
) -> dict[str, object]:
    """Check all expected files and event counts without loading measurements."""

    reports: list[dict[str, object]] = []
    issues: list[str] = []
    component_names = ("eeg_set", "eeg_fdt", "emg_set", "emg_fdt", "annotation")

    for item in registry:
        report: dict[str, object] = {
            "subject": item.subject,
            "condition": item.condition,
            "event_code": item.event_code,
        }
        item_issues: list[str] = []
        files: dict[str, dict[str, object]] = {}
        for component in component_names:
            path = getattr(item, component)
            present = path.is_file()
            file_report: dict[str, object] = {
                "path": str(path),
                "present": present,
            }
            if present:
                file_report["size_bytes"] = path.stat().st_size
            else:
                item_issues.append(f"Missing {component}: {path.name}")
            files[component] = file_report

        annotation_counts: dict[str, int] = {}
        trial_durations_s: list[float] = []
        orphan_starts = 0
        orphan_stops = 0
        if item.annotation.is_file():
            try:
                events = _annotation_events(item.annotation)
                annotation_counts = _annotation_counts(events)
                trial_durations_s, orphan_starts, orphan_stops = _pair_trial_events(
                    events,
                    start_annotation=start_annotation,
                    stop_annotation=stop_annotation,
                )
                start_count = annotation_counts.get(start_annotation, 0)
                if start_count != expected_epoch_count:
                    item_issues.append(
                        f"Expected {expected_epoch_count} '{start_annotation}' annotations; "
                        f"found {start_count}."
                    )
                if len(trial_durations_s) != expected_epoch_count:
                    item_issues.append(
                        f"Expected {expected_epoch_count} paired trial intervals; "
                        f"found {len(trial_durations_s)}."
                    )
                bad_durations = [
                    duration
                    for duration in trial_durations_s
                    if abs(duration - expected_trial_duration_s)
                    > trial_duration_tolerance_s
                ]
                if bad_durations:
                    item_issues.append(
                        "Trial durations outside tolerance: "
                        + ", ".join(f"{duration:.6g}" for duration in bad_durations)
                    )
                if orphan_starts:
                    item_issues.append(f"Found {orphan_starts} unpaired trial start(s).")
            except Exception as error:
                item_issues.append(
                    f"Could not parse annotation: {type(error).__name__}: {error}"
                )

        report["files"] = files
        report["annotation_counts"] = {
            start_annotation: annotation_counts.get(start_annotation, 0),
            stop_annotation: annotation_counts.get(stop_annotation, 0),
        }
        report["paired_trial_count"] = len(trial_durations_s)
        report["trial_durations_s"] = trial_durations_s
        report["orphan_start_count"] = orphan_starts
        report["ignored_pretrial_stop_count"] = orphan_stops
        report["status"] = "pass" if not item_issues else "fail"
        report["issues"] = item_issues
        issues.extend(
            f"{item.subject}/{item.condition}: {message}" for message in item_issues
        )
        reports.append(report)

    return {
        "status": "pass" if not issues else "fail",
        "dataset_grain": "one aligned EEG/EMG EEGLAB pair per subject and condition",
        "expected_dataset_count": len(registry),
        "passed_dataset_count": sum(report["status"] == "pass" for report in reports),
        "expected_files_per_dataset": len(component_names),
        "start_annotation": start_annotation,
        "stop_annotation": stop_annotation,
        "expected_epoch_count": expected_epoch_count,
        "expected_trial_duration_s": expected_trial_duration_s,
        "trial_duration_tolerance_s": trial_duration_tolerance_s,
        "issues": issues,
        "datasets": reports,
    }


def read_eeglab_header(path: str | Path) -> dict[str, object]:
    """Read structural EEGLAB metadata without loading the external FDT signal."""

    source = Path(path)
    names = [
        "nbchan",
        "trials",
        "pnts",
        "srate",
        "xmin",
        "xmax",
        "data",
        "ref",
        "icaweights",
        "chanlocs",
    ]
    mat = loadmat(
        source,
        variable_names=names,
        struct_as_record=False,
        squeeze_me=True,
    )

    def scalar(name: str, cast: type[int] | type[float]) -> int | float:
        if name not in mat:
            raise ValueError(f"EEGLAB header is missing {name!r}")
        return cast(np.asarray(mat[name]).item())

    locations = np.atleast_1d(mat.get("chanlocs", np.array([], dtype=object)))
    channel_names = [str(getattr(location, "labels", "")) for location in locations]
    weights = np.asarray(mat.get("icaweights", np.empty((0, 0))))
    if weights.size == 0:
        ica_component_count = 0
    elif weights.ndim == 1:
        ica_component_count = 1
    else:
        ica_component_count = int(weights.shape[0])

    return {
        "file": str(source),
        "nbchan": scalar("nbchan", int),
        "trials": scalar("trials", int),
        "pnts": scalar("pnts", int),
        "srate": scalar("srate", float),
        "xmin": scalar("xmin", float),
        "xmax": scalar("xmax", float),
        "external_data_file": str(mat.get("data", "")),
        "reference": str(mat.get("ref", "")),
        "ica_component_count": ica_component_count,
        "channel_names": channel_names,
    }


def profile_neuro_headers(
    registry: Sequence[NeuroInput],
    *,
    expected_epoch_count: int = 5,
    expected_sampling_rate_hz: float = 1000.0,
    expected_sample_count: int = 60_000,
    expected_eeg_channel_count: int = 61,
    expected_emg_channel_count: int = 1,
    required_eeg_channels: Sequence[str] = (),
) -> dict[str, object]:
    """Validate EEGLAB dimensions and EEG/EMG alignment for every registry row."""

    reports: list[dict[str, object]] = []
    issues: list[str] = []
    warnings: list[str] = []
    for item in registry:
        item_issues: list[str] = []
        item_warnings: list[str] = []
        report: dict[str, object] = {
            "subject": item.subject,
            "condition": item.condition,
            "event_code": item.event_code,
        }
        try:
            eeg = read_eeglab_header(item.eeg_set)
            emg = read_eeglab_header(item.emg_set)
            report["eeg"] = eeg
            report["emg"] = emg

            expectations = (
                ("EEG epoch count", eeg["trials"], expected_epoch_count),
                ("EMG epoch count", emg["trials"], expected_epoch_count),
                ("EEG sample count", eeg["pnts"], expected_sample_count),
                ("EMG sample count", emg["pnts"], expected_sample_count),
                ("EEG channel count", eeg["nbchan"], expected_eeg_channel_count),
                ("EMG channel count", emg["nbchan"], expected_emg_channel_count),
            )
            for label, actual, expected in expectations:
                if actual != expected:
                    item_issues.append(f"{label} is {actual}; expected {expected}.")
            for label, actual in (
                ("EEG sampling rate", eeg["srate"]),
                ("EMG sampling rate", emg["srate"]),
            ):
                if not np.isclose(actual, expected_sampling_rate_hz):
                    item_issues.append(
                        f"{label} is {actual}; expected {expected_sampling_rate_hz}."
                    )
            for field in ("trials", "pnts", "srate", "xmin", "xmax"):
                if not np.isclose(eeg[field], emg[field]):
                    item_issues.append(
                        f"EEG/EMG {field} mismatch: {eeg[field]} versus {emg[field]}."
                    )
            for label, header, expected_file in (
                ("EEG", eeg, item.eeg_fdt.name),
                ("EMG", emg, item.emg_fdt.name),
            ):
                if header["external_data_file"] != expected_file:
                    item_warnings.append(
                        f"{label} header references {header['external_data_file']!r}; "
                        f"reader uses present sibling {expected_file!r}."
                    )
            missing_roi = sorted(
                set(required_eeg_channels) - set(eeg["channel_names"])
            )
            if missing_roi:
                item_issues.append(f"Missing required EEG channels: {missing_roi}")
        except Exception as error:
            item_issues.append(f"{type(error).__name__}: {error}")

        report["status"] = "pass" if not item_issues else "fail"
        report["issues"] = item_issues
        report["warnings"] = item_warnings
        issues.extend(
            f"{item.subject}/{item.condition}: {message}" for message in item_issues
        )
        warnings.extend(
            f"{item.subject}/{item.condition}: {message}" for message in item_warnings
        )
        reports.append(report)

    return {
        "status": "pass" if not issues else "fail",
        "expected_dataset_count": len(registry),
        "passed_dataset_count": sum(report["status"] == "pass" for report in reports),
        "issues": issues,
        "warnings": warnings,
        "datasets": reports,
    }

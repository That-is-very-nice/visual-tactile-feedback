"""Inspectable quality checks for the repaired force-sensor dataset."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .behavior import read_force_csv, sampling_rate_hz, slice_trial, trial_windows


def profile_behavior_dataset(
    *,
    data_dir: str | Path,
    subjects: Sequence[str],
    conditions: Sequence[str],
    trial_indices: Sequence[int],
    start_offsets_ms: Mapping[str, float],
    file_template: str = "{subject}_{condition}.csv",
    subject_subdirectories: Mapping[str, str] | None = None,
    expected_trial_count: int | None = 5,
    expected_sampling_rate_hz: float | None = 500.0,
    sampling_rate_tolerance_hz: float = 1e-6,
    compute_sha256: bool = False,
) -> dict[str, object]:
    """Profile all configured files without exposing force measurements.

    The report contains only file names, structural counts, sampling rates, and
    validation messages. It is therefore useful as an audit artifact while the
    original human-participant measurements remain outside the repository.
    """

    root = Path(data_dir)
    requested_trials = tuple(int(index) for index in trial_indices)
    expected_names: set[str] = set()
    file_reports: list[dict[str, object]] = []
    issues: list[str] = []

    for subject in subjects:
        for condition in conditions:
            subject_dir = (
                subject_subdirectories.get(subject, subject)
                if subject_subdirectories is not None
                else ""
            )
            relative_path = Path(
                file_template.format(
                    subject=subject,
                    condition=condition,
                    subject_dir=subject_dir,
                )
            )
            path = root / relative_path
            expected_names.add(relative_path.as_posix())
            report: dict[str, object] = {
                "file": relative_path.as_posix(),
                "subject": subject,
                "condition": condition,
                "present": path.is_file(),
            }
            if not path.is_file():
                message = f"Missing configured file: {relative_path.as_posix()}"
                report["issues"] = [message]
                issues.append(message)
                file_reports.append(report)
                continue

            file_issues: list[str] = []
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    frame = read_force_csv(path)
                windows = trial_windows(
                    frame,
                    start_offset_ms=float(start_offsets_ms.get(condition, 0.0)),
                )
                if expected_trial_count is not None and len(windows) != expected_trial_count:
                    file_issues.append(
                        f"Expected {expected_trial_count} trials; found {len(windows)}."
                    )

                windows_by_index = {window.trial_index: window for window in windows}
                sample_counts: list[int] = []
                sampling_rates: list[float] = []
                for trial_index in requested_trials:
                    if trial_index not in windows_by_index:
                        file_issues.append(f"Missing analyzed trial {trial_index}.")
                        continue
                    time_ms, _, _ = slice_trial(frame, windows_by_index[trial_index])
                    if not np.all(np.diff(time_ms) > 0):
                        file_issues.append(
                            f"Trial {trial_index} timestamps are not strictly increasing."
                        )
                        continue
                    rate = sampling_rate_hz(time_ms)
                    sample_counts.append(int(time_ms.size))
                    sampling_rates.append(float(rate))
                    if (
                        expected_sampling_rate_hz is not None
                        and abs(rate - expected_sampling_rate_hz)
                        > sampling_rate_tolerance_hz
                    ):
                        file_issues.append(
                            f"Trial {trial_index} sampling rate is {rate:.9g} Hz; "
                            f"expected {expected_sampling_rate_hz:.9g} Hz."
                        )

                report.update(
                    {
                        "size_bytes": path.stat().st_size,
                        "row_count": int(len(frame)),
                        "trial_count": int(len(windows)),
                        "analyzed_trial_count": int(len(sample_counts)),
                        "analyzed_samples_min": min(sample_counts) if sample_counts else None,
                        "analyzed_samples_max": max(sample_counts) if sample_counts else None,
                        "sampling_rate_hz_min": min(sampling_rates)
                        if sampling_rates
                        else None,
                        "sampling_rate_hz_max": max(sampling_rates)
                        if sampling_rates
                        else None,
                        "reader_warning_count": len(captured),
                    }
                )
                if compute_sha256:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    report["sha256"] = digest.hexdigest()
                if captured:
                    file_issues.append(
                        f"Reader emitted {len(captured)} data-cleaning warning(s)."
                    )
            except Exception as error:  # report all configured files in one pass
                file_issues.append(f"{type(error).__name__}: {error}")

            report["issues"] = file_issues
            issues.extend(f"{relative_path.as_posix()}: {item}" for item in file_issues)
            file_reports.append(report)

    extra_files: list[str] = []
    if root.is_dir():
        discovered = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*.csv")
            if not path.name.startswith("._")
        }
        extra_files = sorted(discovered - expected_names)

    checked = [report for report in file_reports if report["present"]]
    row_counts = [int(report["row_count"]) for report in checked if "row_count" in report]
    sample_mins = [
        int(report["analyzed_samples_min"])
        for report in checked
        if report.get("analyzed_samples_min") is not None
    ]
    sample_maxes = [
        int(report["analyzed_samples_max"])
        for report in checked
        if report.get("analyzed_samples_max") is not None
    ]
    rate_mins = [
        float(report["sampling_rate_hz_min"])
        for report in checked
        if report.get("sampling_rate_hz_min") is not None
    ]
    rate_maxes = [
        float(report["sampling_rate_hz_max"])
        for report in checked
        if report.get("sampling_rate_hz_max") is not None
    ]

    return {
        "status": "pass" if not issues else "fail",
        "dataset_grain": "one force CSV per subject and condition",
        "expected_file_count": len(file_reports),
        "present_file_count": len(checked),
        "checked_file_count": sum("row_count" in report for report in checked),
        "extra_csv_files": extra_files,
        "summary": {
            "row_count_min": min(row_counts) if row_counts else None,
            "row_count_max": max(row_counts) if row_counts else None,
            "analyzed_samples_per_trial_min": min(sample_mins) if sample_mins else None,
            "analyzed_samples_per_trial_max": max(sample_maxes) if sample_maxes else None,
            "sampling_rate_hz_min": min(rate_mins) if rate_mins else None,
            "sampling_rate_hz_max": max(rate_maxes) if rate_maxes else None,
        },
        "issues": issues,
        "files": file_reports,
    }

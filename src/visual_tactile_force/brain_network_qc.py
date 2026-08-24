"""Metadata-only quality checks for brain-network EEG inputs."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .neuro_registry import NeuroInput, read_eeglab_header


def profile_brain_network_inputs(
    registry: Sequence[NeuroInput],
    *,
    expected_epoch_count: int,
    expected_sampling_rate_hz: float,
    expected_sample_count: int,
    expected_eeg_channel_count: int,
    required_eeg_channels: Sequence[str],
) -> dict[str, object]:
    """Check only EEG files and headers because this pipeline never reads EMG."""

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
            "eeg_set": str(item.eeg_set),
            "eeg_fdt": str(item.eeg_fdt),
        }
        for label, path in (("EEG set", item.eeg_set), ("EEG fdt", item.eeg_fdt)):
            if not path.is_file():
                item_issues.append(f"Missing {label}: {path.name}")
        if item.eeg_set.is_file():
            try:
                header = read_eeglab_header(item.eeg_set)
                report["header"] = header
                expectations = (
                    ("epoch count", header["trials"], expected_epoch_count),
                    ("sample count", header["pnts"], expected_sample_count),
                    ("EEG channel count", header["nbchan"], expected_eeg_channel_count),
                )
                for label, actual, expected in expectations:
                    if actual != expected:
                        item_issues.append(f"{label} is {actual}; expected {expected}.")
                if not np.isclose(header["srate"], expected_sampling_rate_hz):
                    item_issues.append(
                        f"sampling rate is {header['srate']}; expected {expected_sampling_rate_hz}."
                    )
                missing_channels = sorted(
                    set(required_eeg_channels) - set(header["channel_names"])
                )
                if missing_channels:
                    item_issues.append(f"Missing required EEG channels: {missing_channels}")
                if header["external_data_file"] != item.eeg_fdt.name:
                    item_warnings.append(
                        "EEG header references "
                        f"{header['external_data_file']!r}; expected sibling {item.eeg_fdt.name!r}."
                    )
            except Exception as error:
                item_issues.append(f"{type(error).__name__}: {error}")
        report["status"] = "pass" if not item_issues else "fail"
        report["issues"] = item_issues
        report["warnings"] = item_warnings
        reports.append(report)
        issues.extend(f"{item.subject}/{item.condition}: {message}" for message in item_issues)
        warnings.extend(
            f"{item.subject}/{item.condition}: {message}" for message in item_warnings
        )
    return {
        "status": "pass" if not issues else "fail",
        "dataset_grain": "one preprocessed EEG EEGLAB pair per subject and condition",
        "expected_dataset_count": len(registry),
        "passed_dataset_count": sum(report["status"] == "pass" for report in reports),
        "issues": issues,
        "warnings": warnings,
        "datasets": reports,
    }

"""Command-line entry point for the sensor-space brain-network analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from . import __version__
from .brain_network import (
    PAPER_BANDS_HZ,
    PAPER_NETWORK_METRIC,
    PAPER_NETWORK_METRIC_DEFINITION,
    PAPER_ROI_CHANNELS,
    mapped_channels,
    validate_bands,
    validate_roi_map,
)
from .brain_network_batch import (
    brain_network_results_to_frame,
    collapse_directed_brain_network_rows,
    summarize_brain_network_wilcoxon_holm,
)
from .brain_network_pipeline import compute_brain_network_trace
from .brain_network_qc import profile_brain_network_inputs
from .brain_network_regression import compare_brain_network_significant_rows
from .brain_network_statistics import load_saved_brain_network_statistics
from .neuro_registry import NeuroInput, build_neuro_registry


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_config(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _validate_network_config(network: Mapping[str, object]) -> None:
    if network.get("estimator") != "multitaper absolute imaginary coherence":
        raise ValueError("Brain-network estimator must be multitaper absolute imaginary coherence")
    if not bool(network.get("apply_csd")):
        raise ValueError("Brain-network paper method requires current-source density")
    if not bool(network.get("include_within_roi")):
        raise ValueError("Brain-network statistics require within-ROI values")
    if str(network.get("summary_metric")) != PAPER_NETWORK_METRIC:
        raise ValueError(f"Stable network metric must be {PAPER_NETWORK_METRIC!r}")

    roi_channels = network.get("roi_channels")
    bands_hz = network.get("bands_hz")
    if not isinstance(roi_channels, Mapping) or not isinstance(bands_hz, Mapping):
        raise ValueError("brain_network.roi_channels and bands_hz must be mappings")
    validate_roi_map(roi_channels)
    validate_bands(bands_hz)
    frozen_rois = {name: tuple(channels) for name, channels in roi_channels.items()}
    if frozen_rois != PAPER_ROI_CHANNELS:
        raise ValueError("Configured ROI map differs from the frozen paper ROI map")
    frozen_bands = {
        name: tuple(float(value) for value in bounds) for name, bounds in bands_hz.items()
    }
    if frozen_bands != PAPER_BANDS_HZ:
        raise ValueError("Configured frequency bands differ from the frozen paper bands")

    statistics = network.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("brain_network.statistics must be a mapping")
    expected_statistics = {
        "test": "wilcoxon",
        "alternative": "two-sided",
        "zero_method": "pratt",
        "continuity_correction": True,
        "method": "auto",
        "correction": "holm",
        "correction_scope": "per_band",
        "directed_roi_pairs": True,
        "include_within_roi": True,
        "family_size": 100,
        "alpha": 0.05,
    }
    for key, expected_value in expected_statistics.items():
        if statistics.get(key) != expected_value:
            raise ValueError(
                f"brain_network.statistics.{key} must be {expected_value!r}"
            )

    trial_indices = list(network["trial_indices"])
    duration = float(network["tmax_s"]) - float(network["tmin_s"])
    segment_duration = float(network["segment_duration_s"])
    calculated = len(trial_indices) * duration / segment_duration
    rounded = int(round(calculated))
    if rounded <= 0 or abs(calculated - rounded) > 1e-9:
        raise ValueError("Brain-network window must contain a whole number of segments")
    if int(network.get("expected_segment_count", rounded)) != rounded:
        raise ValueError("expected_segment_count does not match network segmentation")


def _build_registry(config: Mapping[str, object]) -> list[NeuroInput]:
    paths, neuro = config["paths"], config["neuro"]
    if not isinstance(paths, Mapping) or not isinstance(neuro, Mapping):
        raise ValueError("paths and neuro must be mappings")
    return build_neuro_registry(
        set_dir=paths["eeglab_set_dir"],
        annotation_dir=paths["annotation_dir"],
        subjects=neuro["subjects"],
        condition_event_codes=neuro["subject_condition_event_codes"],
        epoch_bounds_label=str(neuro.get("epoch_bounds_label", "[0 60]")),
    )


def _quality_report(
    config: Mapping[str, object], registry: Sequence[NeuroInput]
) -> dict[str, object]:
    neuro, network = config["neuro"], config["brain_network"]
    if not isinstance(neuro, Mapping) or not isinstance(network, Mapping):
        raise ValueError("neuro and brain_network must be mappings")
    roi_channels = network["roi_channels"]
    if not isinstance(roi_channels, Mapping):
        raise ValueError("brain_network.roi_channels must be a mapping")
    sampling_rate = float(network["expected_sampling_rate_hz"])
    return profile_brain_network_inputs(
        registry,
        expected_epoch_count=int(neuro.get("expected_epoch_count", 5)),
        expected_sampling_rate_hz=sampling_rate,
        expected_sample_count=int(round(float(network["tmax_s"]) * sampling_rate)),
        expected_eeg_channel_count=int(neuro.get("expected_eeg_channel_count", 61)),
        required_eeg_channels=mapped_channels(roi_channels),
    )


def _write_quality_outputs(
    output_dir: Path,
    registry: Sequence[NeuroInput],
    report: Mapping[str, object],
) -> None:
    _write_json(output_dir / "brain_network_input_quality.json", report)
    pd.DataFrame(item.to_record() for item in registry).to_csv(
        output_dir / "brain_network_input_registry.csv", index=False
    )


def _raise_for_quality(report: Mapping[str, object]) -> None:
    if report.get("status") != "pass":
        preview = "; ".join(str(issue) for issue in list(report.get("issues", []))[:5])
        raise ValueError(f"Brain-network input quality checks failed: {preview}")


def run_network_qc(config_path: Path, output_dir: Path) -> int:
    config = _read_config(config_path)
    network = config["brain_network"]
    if not isinstance(network, Mapping):
        raise ValueError("brain_network must be a mapping")
    _validate_network_config(network)
    registry = _build_registry(config)
    report = _quality_report(config, registry)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_quality_outputs(output_dir, registry, report)
    _raise_for_quality(report)
    return 0


def _compute_item(
    item: NeuroInput,
    network: Mapping[str, object],
    *,
    n_jobs: int,
) -> dict[str, object]:
    roi_channels = network["roi_channels"]
    bands_hz = network["bands_hz"]
    if not isinstance(roi_channels, Mapping) or not isinstance(bands_hz, Mapping):
        raise ValueError("ROI channels and bands must be mappings")
    return compute_brain_network_trace(
        item,
        trial_indices=network["trial_indices"],
        tmin_s=float(network["tmin_s"]),
        tmax_s=float(network["tmax_s"]),
        segment_duration_s=float(network["segment_duration_s"]),
        apply_csd=bool(network["apply_csd"]),
        csd_stiffness=float(network["csd_stiffness"]),
        csd_n_legendre_terms=int(network["csd_n_legendre_terms"]),
        csd_lambda2=float(network["csd_lambda2"]),
        roi_channels=roi_channels,
        bands_hz=bands_hz,
        multitaper_bandwidth_hz=float(network["multitaper_bandwidth_hz"]),
        multitaper_adaptive=bool(network["multitaper_adaptive"]),
        include_within_roi=bool(network["include_within_roi"]),
        n_jobs=n_jobs,
    )


def _expected_file(
    config_path: Path,
    regression: Mapping[str, object],
    field: str,
) -> tuple[Path, dict[str, object], float]:
    name = regression.get(field)
    if not name:
        raise ValueError(f"brain_network_regression.{field} is required")
    path = Path(str(name))
    if not path.is_absolute():
        path = config_path.parent / path
    expected = json.loads(path.read_text(encoding="utf-8"))
    tolerance = float(
        regression.get("absolute_tolerance", expected.get("absolute_tolerance", 1e-12))
    )
    return path, expected, tolerance


def _statistics_regression(
    config_path: Path,
    config: Mapping[str, object],
    directed_rows: Sequence[Mapping[str, object]],
    canonical_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    regression = config.get("brain_network_regression", {})
    if not isinstance(regression, Mapping):
        raise ValueError("brain_network_regression must be a mapping")

    path, expected, tolerance = _expected_file(
        config_path, regression, "expected_statistics_file"
    )
    report = compare_brain_network_significant_rows(
        canonical_rows,
        expected,
        p_column="p_value_holm",
        absolute_tolerance=tolerance,
    )
    expected_directed_count = int(expected.get("expected_directed_row_count", 500))
    actual_directed_count = len(directed_rows)
    expected_significant_directed = int(
        expected.get("expected_significant_directed_count", 11)
    )
    actual_significant_directed = sum(
        float(row["p_value_holm"]) < 0.05 for row in directed_rows
    )
    if actual_directed_count != expected_directed_count:
        report["failures"].append(
            "directed row count differs: "
            f"actual={actual_directed_count}, expected={expected_directed_count}"
        )
    if actual_significant_directed != expected_significant_directed:
        report["failures"].append(
            "significant directed row count differs: "
            f"actual={actual_significant_directed}, "
            f"expected={expected_significant_directed}"
        )
    report["actual_directed_row_count"] = actual_directed_count
    report["expected_directed_row_count"] = expected_directed_count
    report["actual_significant_directed_count"] = actual_significant_directed
    report["expected_significant_directed_count"] = expected_significant_directed
    report["status"] = "pass" if not report["failures"] else "fail"
    report["expected_statistics_file"] = path.name
    return report


def _statistics_validation(
    directed_rows: Sequence[Mapping[str, object]],
    canonical_rows: Sequence[Mapping[str, object]],
    *,
    band_order: Sequence[str],
) -> dict[str, object]:
    """Validate the fixed row structure produced by the statistical pipeline."""

    directed = pd.DataFrame(directed_rows)
    canonical = pd.DataFrame(canonical_rows)
    expected_directed = {str(band): 100 for band in band_order}
    expected_canonical = {str(band): 55 for band in band_order}
    actual_directed = {
        str(key): int(value) for key, value in directed.groupby("band").size().items()
    }
    actual_canonical = {
        str(key): int(value) for key, value in canonical.groupby("band").size().items()
    }
    failures: list[str] = []
    if actual_directed != expected_directed:
        failures.append(
            f"directed rows per band differ: actual={actual_directed}, "
            f"expected={expected_directed}"
        )
    if actual_canonical != expected_canonical:
        failures.append(
            f"canonical rows per band differ: actual={actual_canonical}, "
            f"expected={expected_canonical}"
        )
    if set(directed["analysis_profile"].astype(str)) != {
        "wilcoxon_holm_per_band_directed_100"
    }:
        failures.append("directed rows contain an unexpected analysis profile")
    if set(directed["correction_family_size"].astype(int)) != {100}:
        failures.append("directed rows contain an unexpected correction family size")
    if directed.duplicated(["band", "roi_source", "roi_target"]).any():
        failures.append("directed statistical rows are not unique")
    if canonical.duplicated(["band", "roi_source", "roi_target"]).any():
        failures.append("canonical statistical rows are not unique")
    return {
        "status": "pass" if not failures else "fail",
        "analysis_profile": "wilcoxon_holm_per_band_directed_100",
        "directed_row_count": int(len(directed)),
        "canonical_row_count": int(len(canonical)),
        "directed_rows_per_band": actual_directed,
        "canonical_rows_per_band": actual_canonical,
        "failures": failures,
    }


def run_network(
    config_path: Path,
    output_dir: Path,
    *,
    n_jobs: int,
    resume: bool,
) -> int:
    """Run QC, |ImCoh|, Wilcoxon-Holm statistics, validation, and plots."""

    config = _read_config(config_path)
    network, neuro = config["brain_network"], config["neuro"]
    if not isinstance(network, Mapping) or not isinstance(neuro, Mapping):
        raise ValueError("brain_network and neuro must be mappings")
    _validate_network_config(network)
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    visual = str(neuro.get("visual_condition", "st_no"))
    tactile = str(neuro.get("tactile_condition", "st_tf2"))

    output_dir.mkdir(parents=True, exist_ok=True)
    registry = _build_registry(config)
    quality = _quality_report(config, registry)
    _write_quality_outputs(output_dir, registry, quality)
    _raise_for_quality(quality)

    runtime_config = {
        "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "brain_network": network,
        "subject_condition_event_codes": neuro["subject_condition_event_codes"],
    }
    config_sha256 = hashlib.sha256(
        json.dumps(runtime_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint_path = output_dir / "brain_network_checkpoint.json"
    results: list[dict[str, object]] = []
    if resume and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("config_sha256") != config_sha256:
            raise ValueError("Checkpoint config hash does not match the requested config")
        results = list(checkpoint.get("results", []))
        if any("roi_connectivity" not in result for result in results):
            raise ValueError("Checkpoint uses an incompatible brain-network schema")
    completed = {(row["subject"], row["condition"]) for row in results}
    for index, item in enumerate(registry, start=1):
        key = (item.subject, item.condition)
        if key in completed:
            print(
                f"Network {index}/{len(registry)}: {item.subject}/{item.condition} cached",
                flush=True,
            )
            continue
        print(f"Network {index}/{len(registry)}: {item.subject}/{item.condition}", flush=True)
        results.append(_compute_item(item, network, n_jobs=n_jobs))
        _write_json(
            checkpoint_path,
            {"status": "in_progress", "config_sha256": config_sha256, "results": results},
        )

    summary = brain_network_results_to_frame(results)
    directed_rows = summarize_brain_network_wilcoxon_holm(
        summary,
        visual_condition=visual,
        tactile_condition=tactile,
    )
    canonical_rows = collapse_directed_brain_network_rows(directed_rows)

    significant_interregional = [
        row
        for row in canonical_rows
        if row["edge_scope"] == "interregional" and float(row["p_value_holm"]) < 0.05
    ]
    summary.to_csv(
        output_dir / "brain_network_subject_summary.csv",
        index=False,
    )
    pd.DataFrame(directed_rows).to_csv(
        output_dir / "brain_network_statistics_all.csv",
        index=False,
    )
    pd.DataFrame(canonical_rows).to_csv(
        output_dir / "brain_network_statistics_canonical.csv",
        index=False,
    )
    pd.DataFrame(significant_interregional).to_csv(
        output_dir / "brain_network_significant_interregional.csv",
        index=False,
    )
    _write_json(
        output_dir / "brain_network_statistics.json",
        {
            "schema_version": 1,
            "analysis_profile": "wilcoxon_holm_per_band_directed_100",
            "metric": PAPER_NETWORK_METRIC,
            "metric_definition": PAPER_NETWORK_METRIC_DEFINITION,
            "difference_definition": f"{visual} minus {tactile}",
            "correction": "Holm step-down within each frequency band",
            "correction_family_size": 100,
            "directed_statistics": directed_rows,
            "canonical_statistics": canonical_rows,
            "significant_interregional": significant_interregional,
        },
    )
    validation = _statistics_validation(
        directed_rows,
        canonical_rows,
        band_order=list(network["bands_hz"]),
    )
    _write_json(output_dir / "brain_network_validation.json", validation)
    if validation["status"] != "pass":
        raise ValueError("Brain-network statistical validation failed")

    from .brain_network_figures import (
        plot_condition_network_matrices,
        plot_difference_matrices,
    )

    band_order = list(network["bands_hz"])
    plot_condition_network_matrices(
        summary,
        visual_condition=visual,
        tactile_condition=tactile,
        band_order=band_order,
        output_paths=(
            output_dir / "figure_9_brain_network.png",
            output_dir / "figure_9_brain_network.pdf",
        ),
    )
    canonical_frame = pd.DataFrame(canonical_rows)
    plot_difference_matrices(
        canonical_frame,
        band_order=band_order,
        p_column="p_value_holm",
        output_paths=(
            output_dir / "figure_10_brain_network.png",
            output_dir / "figure_10_brain_network.pdf",
        ),
    )

    import mne
    import mne_connectivity
    import numpy as np
    import scipy

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "sensor_space_absolute_imcoh_roi_network",
        "pipeline_version": __version__,
        "randomness": "none",
        "statistics": {
            "test": "paired two-sided Wilcoxon signed-rank",
            "zero_method": "pratt",
            "continuity_correction": True,
            "correction": "Holm step-down",
            "correction_scope": "per frequency band",
            "correction_family_size": 100,
        },
        "config": {
            "file": config_path.name,
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "runtime_sha256": config_sha256,
            "brain_network": network,
        },
        "data": {
            "quality_status": quality["status"],
            "dataset_count": len(registry),
            "datasets": [
                {
                    "subject": item.subject,
                    "condition": item.condition,
                    "event_code": item.event_code,
                    "eeg_set": item.eeg_set.name,
                    "eeg_fdt": item.eeg_fdt.name,
                }
                for item in registry
            ],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "mne": mne.__version__,
            "mne_connectivity": mne_connectivity.__version__,
        },
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    _write_json(output_dir / "brain_network_manifest.json", manifest)
    _write_json(
        checkpoint_path,
        {"status": "complete", "config_sha256": config_sha256, "results": results},
    )
    return 0


def run_verify_results(config_path: Path, output_dir: Path) -> int:
    """Verify the saved complete brain-network statistical output."""

    config = _read_config(config_path)
    paths = config["paths"]
    if not isinstance(paths, Mapping):
        raise ValueError("paths must be a mapping")
    statistics_path = Path(str(paths["brain_network_statistics_csv"]))
    directed_frame = load_saved_brain_network_statistics(statistics_path)
    canonical_rows = collapse_directed_brain_network_rows(
        directed_frame.to_dict(orient="records")
    )
    significant_interregional = [
        row
        for row in canonical_rows
        if row["edge_scope"] == "interregional" and float(row["p_value_holm"]) < 0.05
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    directed_frame.to_csv(output_dir / "brain_network_statistics_all.csv", index=False)
    pd.DataFrame(canonical_rows).to_csv(
        output_dir / "brain_network_statistics_canonical.csv", index=False
    )
    pd.DataFrame(significant_interregional).to_csv(
        output_dir / "brain_network_significant_interregional.csv", index=False
    )
    report = _statistics_regression(
        config_path,
        config,
        directed_frame.to_dict(orient="records"),
        canonical_rows,
    )
    report["statistics_csv_sha256"] = hashlib.sha256(statistics_path.read_bytes()).hexdigest()
    _write_json(output_dir / "brain_network_regression.json", report)
    if report["status"] != "pass":
        raise ValueError("Brain-network result verification failed")

    from .brain_network_figures import plot_difference_matrices

    plot_difference_matrices(
        pd.DataFrame(canonical_rows),
        band_order=list(PAPER_BANDS_HZ),
        p_column="p_value_holm",
        output_paths=(output_dir / "figure_10_brain_network.png",),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("qc", "validate EEG files and headers without loading signal data"),
        ("run", "run the brain-network analysis"),
        ("verify", "verify saved brain-network results"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "run":
            child.add_argument("--n-jobs", type=int, default=1)
            child.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "qc":
            return run_network_qc(arguments.config, arguments.output_dir)
        if arguments.command == "verify":
            return run_verify_results(arguments.config, arguments.output_dir)
        return run_network(
            arguments.config,
            arguments.output_dir,
            n_jobs=arguments.n_jobs,
            resume=arguments.resume,
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

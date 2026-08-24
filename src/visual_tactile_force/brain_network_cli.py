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
    summarize_declared_holm,
    summarize_published_style_max_t,
)
from .brain_network_pipeline import compute_brain_network_trace
from .brain_network_qc import profile_brain_network_inputs
from .brain_network_regression import compare_brain_network_significant_rows
from .legacy_brain_network import load_legacy_roi_connectivity, load_published_max_t
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
    if not bool(network.get("include_within_roi_for_max_t")):
        raise ValueError("Published-style max-T reconstruction requires within-ROI values")
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
        include_within_roi=bool(network["include_within_roi_for_max_t"]),
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


def _corrected_regression(
    config_path: Path,
    config: Mapping[str, object],
    holm_rows: Sequence[Mapping[str, object]],
    max_t_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    regression = config.get("brain_network_regression", {})
    if not isinstance(regression, Mapping):
        raise ValueError("brain_network_regression must be a mapping")
    path, expected, tolerance = _expected_file(
        config_path, regression, "corrected_expected_statistics_file"
    )
    holm_expected = expected.get("declared_holm")
    max_t_expected = expected.get("published_style_max_t")
    if not isinstance(holm_expected, Mapping) or not isinstance(max_t_expected, Mapping):
        raise ValueError("Corrected network baseline must contain both statistical profiles")
    report = {
        "status": "pass",
        "expected_statistics_file": path.name,
        "declared_holm": compare_brain_network_significant_rows(
            holm_rows,
            holm_expected,
            p_column="p_value_holm",
            absolute_tolerance=tolerance,
        ),
        "published_style_max_t": compare_brain_network_significant_rows(
            max_t_rows,
            max_t_expected,
            p_column="p_value_max_t",
            absolute_tolerance=tolerance,
        ),
    }
    if any(report[name]["status"] != "pass" for name in ("declared_holm", "published_style_max_t")):
        report["status"] = "fail"
    return report


def run_network(
    config_path: Path,
    output_dir: Path,
    *,
    n_jobs: int,
    resume: bool,
) -> int:
    """Run QC, |ImCoh|, both statistical profiles, plots, and provenance."""

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
    holm_rows = summarize_declared_holm(
        summary, visual_condition=visual, tactile_condition=tactile
    )
    max_t_rows = summarize_published_style_max_t(
        summary, visual_condition=visual, tactile_condition=tactile
    )
    summary.to_csv(output_dir / "brain_network_subject_summary.csv", index=False)
    pd.DataFrame(holm_rows).to_csv(
        output_dir / "brain_network_statistics_declared_holm.csv", index=False
    )
    pd.DataFrame(max_t_rows).to_csv(
        output_dir / "brain_network_statistics_published_style_max_t.csv", index=False
    )
    _write_json(
        output_dir / "brain_network_statistics.json",
        {
            "schema_version": 1,
            "metric": PAPER_NETWORK_METRIC,
            "metric_definition": PAPER_NETWORK_METRIC_DEFINITION,
            "difference_definition": f"{visual} minus {tactile}",
            "declared_holm": holm_rows,
            "published_style_max_t": max_t_rows,
        },
    )
    regression = _corrected_regression(config_path, config, holm_rows, max_t_rows)
    _write_json(output_dir / "brain_network_method_regression.json", regression)
    if regression["status"] != "pass":
        raise ValueError("Brain-network method regression failed; inspect the regression report")

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
            output_dir / "figure_9_brain_network_corrected.png",
            output_dir / "figure_9_brain_network_corrected.pdf",
        ),
    )
    max_t_frame = pd.DataFrame(max_t_rows)
    plot_difference_matrices(
        max_t_frame,
        band_order=band_order,
        p_column="p_value_max_t",
        output_paths=(
            output_dir / "figure_10_brain_network_corrected.png",
            output_dir / "figure_10_brain_network_corrected.pdf",
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
        "randomness": "none; max-T uses all sign patterns",
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


def run_legacy_regression(config_path: Path, output_dir: Path) -> int:
    """Reproduce the published max-T edge set from archived historical CSVs."""

    config = _read_config(config_path)
    paths, neuro = config["paths"], config["neuro"]
    if not isinstance(paths, Mapping) or not isinstance(neuro, Mapping):
        raise ValueError("paths and neuro must be mappings")
    roi_path = Path(str(paths["brain_network_legacy_roi_csv"]))
    max_t_path = Path(str(paths["brain_network_legacy_max_t_csv"]))
    summary = load_legacy_roi_connectivity(roi_path)
    published = load_published_max_t(max_t_path)
    visual = str(neuro.get("visual_condition", "st_no"))
    tactile = str(neuro.get("tactile_condition", "st_tf2"))
    holm_rows = summarize_declared_holm(
        summary, visual_condition=visual, tactile_condition=tactile
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "legacy_brain_network_subject_summary.csv", index=False)
    pd.DataFrame(holm_rows).to_csv(output_dir / "legacy_declared_holm.csv", index=False)
    published.to_csv(output_dir / "published_legacy_max_t.csv", index=False)

    regression_config = config.get("brain_network_regression", {})
    if not isinstance(regression_config, Mapping):
        raise ValueError("brain_network_regression must be a mapping")
    expected_path, expected, tolerance = _expected_file(
        config_path, regression_config, "published_expected_statistics_file"
    )
    report = compare_brain_network_significant_rows(
        published.to_dict(orient="records"),
        expected,
        p_column="p_value_max_t",
        absolute_tolerance=tolerance,
    )
    report["expected_statistics_file"] = expected_path.name
    report["legacy_roi_csv_sha256"] = hashlib.sha256(roi_path.read_bytes()).hexdigest()
    report["legacy_max_t_csv_sha256"] = hashlib.sha256(max_t_path.read_bytes()).hexdigest()
    _write_json(output_dir / "published_legacy_regression.json", report)
    if report["status"] != "pass":
        raise ValueError("Published legacy brain-network regression failed")

    from .brain_network_figures import (
        plot_condition_network_matrices,
        plot_difference_matrices,
    )

    band_order = list(PAPER_BANDS_HZ)
    plot_condition_network_matrices(
        summary,
        visual_condition=visual,
        tactile_condition=tactile,
        band_order=band_order,
        output_paths=(output_dir / "figure_9_published_legacy.png",),
    )
    plot_difference_matrices(
        published,
        band_order=band_order,
        p_column="p_value_max_t",
        output_paths=(output_dir / "figure_10_published_legacy.png",),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("qc", "validate EEG files and headers without loading signal data"),
        ("run", "run corrected brain-network analysis"),
        ("legacy-regression", "reproduce the published historical max-T edge set"),
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
        if arguments.command == "legacy-regression":
            return run_legacy_regression(arguments.config, arguments.output_dir)
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

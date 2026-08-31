"""Command-line entry point for the corrected PDC analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from . import __version__
from .neuro_registry import (
    NeuroInput,
    build_neuro_registry,
    profile_neuro_headers,
    profile_neuro_inputs,
)
from .pdc import (
    PAPER_PDC_METRIC,
    PDC_METRIC_DEFINITIONS,
    PDC_DIRECTIONS,
    monte_carlo_pdc_thresholds,
    pdc_frequency_axis,
)
from .pdc_batch import (
    correlate_pdc_with_behavior,
    paper_pdc_results_to_frame,
    summarize_paper_pdc,
)
from .pdc_pipeline import compute_pdc_trace
from .pdc_regression import compare_pdc_statistics


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_config(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _validate_pdc_config(pdc: Mapping[str, object]) -> None:
    required_literals = {
        "emg_filter_after_alignment": "none",
        "estimator": "bivariate VAR partial directed coherence (SCoT)",
        "model_order_policy": "fixed_explicit",
        "frequency_axis": "0_to_nyquist",
        "null_threshold_type": "frequency_resolved",
    }
    for field, expected in required_literals.items():
        if pdc.get(field) != expected:
            raise ValueError(f"PDC {field} must be {expected!r}")
    if bool(pdc.get("emg_rectification")):
        raise ValueError("PDC requires unrectified EMG")
    if not bool(pdc.get("apply_csd")):
        raise ValueError("PDC paper method requires current-source density")
    metric = str(pdc.get("summary_metric", PAPER_PDC_METRIC))
    supported_metrics = {
        "max_normalized_suprathreshold_excess",  # PDC4
        "max_mean_suprathreshold_excess",        # PDC5
    }
    if metric not in supported_metrics:
        raise ValueError(
            f"Unsupported PDC summary metric {metric!r}; "
            f"expected one of {sorted(supported_metrics)}"
        )

    trial_indices = list(pdc["trial_indices"])
    duration = float(pdc["tmax_s"]) - float(pdc["tmin_s"])
    segment_duration = float(pdc["segment_duration_s"])
    calculated = len(trial_indices) * duration / segment_duration
    rounded = int(round(calculated))
    if rounded <= 0 or abs(calculated - rounded) > 1e-9:
        raise ValueError("PDC analysis window must contain a whole number of segments")
    if int(pdc.get("expected_segment_count", rounded)) != rounded:
        raise ValueError("expected_segment_count does not match PDC segmentation")

    target_rate = float(pdc["target_sampling_rate_hz"])
    samples = target_rate * segment_duration
    if not np.isclose(samples, round(samples), rtol=0.0, atol=1e-9):
        raise ValueError("target sampling rate produces a non-integral segment length")
    if int(pdc["model_order"]) >= int(round(samples)):
        raise ValueError("PDC model_order must be smaller than samples per segment")
    frequencies = pdc_frequency_axis(
        sampling_rate_hz=target_rate,
        n_frequency_bins=int(pdc["n_frequency_bins"]),
    )
    spectral = [float(value) for value in pdc["spectral_range_hz"]]
    if len(spectral) != 2 or not 0 <= spectral[0] < spectral[1] <= frequencies[-1]:
        raise ValueError("spectral_range_hz must lie between zero and Nyquist")
    if int(pdc["monte_carlo_iterations"]) < 1:
        raise ValueError("monte_carlo_iterations must be positive")
    if not 0 < float(pdc["monte_carlo_percentile"]) < 100:
        raise ValueError("monte_carlo_percentile must lie between zero and 100")


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


def _quality_reports(
    config: Mapping[str, object], registry: Sequence[NeuroInput]
) -> tuple[dict[str, object], dict[str, object]]:
    neuro, pdc = config["neuro"], config["pdc"]
    if not isinstance(neuro, Mapping) or not isinstance(pdc, Mapping):
        raise ValueError("neuro and pdc must be mappings")
    input_report = profile_neuro_inputs(
        registry,
        start_annotation=str(neuro.get("start_annotation", "DC trigger 13")),
        stop_annotation=str(neuro.get("stop_annotation", "DC trigger 14")),
        expected_epoch_count=int(neuro.get("expected_epoch_count", 5)),
        expected_trial_duration_s=float(neuro.get("expected_trial_duration_s", 60.0)),
        trial_duration_tolerance_s=float(neuro.get("trial_duration_tolerance_s", 0.1)),
    )
    header_report = profile_neuro_headers(
        registry,
        expected_epoch_count=int(neuro.get("expected_epoch_count", 5)),
        expected_sampling_rate_hz=float(pdc["expected_sampling_rate_hz"]),
        expected_sample_count=int(
            round(float(pdc["tmax_s"]) * float(pdc["expected_sampling_rate_hz"]))
        ),
        expected_eeg_channel_count=int(neuro.get("expected_eeg_channel_count", 61)),
        expected_emg_channel_count=int(neuro.get("expected_emg_channel_count", 1)),
        required_eeg_channels=pdc["roi_channels"],
    )
    return input_report, header_report


def _raise_for_quality(
    input_report: Mapping[str, object], header_report: Mapping[str, object]
) -> None:
    if input_report.get("status") == "pass" and header_report.get("status") == "pass":
        return
    issues = list(input_report.get("issues", [])) + list(header_report.get("issues", []))
    raise ValueError(f"PDC input quality checks failed: {'; '.join(map(str, issues[:5]))}")


def _write_quality_outputs(
    output_dir: Path,
    registry: Sequence[NeuroInput],
    input_report: Mapping[str, object],
    header_report: Mapping[str, object],
) -> None:
    _write_json(output_dir / "pdc_input_quality.json", input_report)
    _write_json(output_dir / "pdc_header_quality.json", header_report)
    pd.DataFrame(item.to_record() for item in registry).to_csv(
        output_dir / "pdc_input_registry.csv", index=False
    )


def run_pdc_qc(config_path: Path, output_dir: Path) -> int:
    """Validate all configured PDC inputs without loading FDT signals."""

    config = _read_config(config_path)
    pdc = config["pdc"]
    if not isinstance(pdc, Mapping):
        raise ValueError("pdc must be a mapping")
    _validate_pdc_config(pdc)
    registry = _build_registry(config)
    input_report, header_report = _quality_reports(config, registry)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_quality_outputs(output_dir, registry, input_report, header_report)
    _raise_for_quality(input_report, header_report)
    return 0


def _threshold_config(pdc: Mapping[str, object]) -> dict[str, object]:
    return {
        "epoch_count": int(pdc["expected_segment_count"]),
        "samples_per_epoch": int(
            round(float(pdc["target_sampling_rate_hz"]) * float(pdc["segment_duration_s"]))
        ),
        "model_order": int(pdc["model_order"]),
        "n_frequency_bins": int(pdc["n_frequency_bins"]),
        "iterations": int(pdc["monte_carlo_iterations"]),
        "percentile": float(pdc["monte_carlo_percentile"]),
        "random_seed": int(pdc["monte_carlo_random_seed"]),
    }


def _load_or_compute_thresholds(
    output_dir: Path,
    pdc: Mapping[str, object],
    *,
    n_jobs: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    settings = _threshold_config(pdc)
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
    array_path = output_dir / "pdc_null_threshold.npz"
    metadata_path = output_dir / "pdc_null_threshold.json"
    if array_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("settings_sha256") == digest:
            with np.load(array_path, allow_pickle=False) as archive:
                thresholds = {direction: archive[direction] for direction in PDC_DIRECTIONS}
            print("PDC Monte Carlo threshold: cached", flush=True)
            return thresholds, metadata

    print(
        f"PDC Monte Carlo threshold: 0/{settings['iterations']} simulations",
        flush=True,
    )

    def progress(done: int, total: int) -> None:
        interval = max(1, total // 20)
        if done == total or done % interval == 0:
            print(f"PDC Monte Carlo threshold: {done}/{total}", flush=True)

    thresholds = monte_carlo_pdc_thresholds(
        **settings,
        n_jobs=n_jobs,
        progress=progress,
    )
    temporary = array_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **thresholds)
    temporary.replace(array_path)
    metadata = {
        "schema_version": 1,
        "settings": settings,
        "settings_sha256": digest,
        "directions": list(PDC_DIRECTIONS),
        "threshold_shape": [int(settings["n_frequency_bins"])],
        "frequency_axis": "0_to_nyquist",
    }
    _write_json(metadata_path, metadata)
    return thresholds, metadata


def _compute_item(
    item: NeuroInput,
    pdc: Mapping[str, object],
    thresholds: Mapping[str, np.ndarray],
    *,
    n_jobs: int,
) -> dict[str, object]:
    return compute_pdc_trace(
        item,
        trial_indices=pdc["trial_indices"],
        tmin_s=float(pdc["tmin_s"]),
        tmax_s=float(pdc["tmax_s"]),
        segment_duration_s=float(pdc["segment_duration_s"]),
        apply_csd=bool(pdc["apply_csd"]),
        csd_stiffness=float(pdc["csd_stiffness"]),
        csd_n_legendre_terms=int(pdc["csd_n_legendre_terms"]),
        csd_lambda2=float(pdc["csd_lambda2"]),
        target_sampling_rate_hz=float(pdc["target_sampling_rate_hz"]),
        model_order=int(pdc["model_order"]),
        n_frequency_bins=int(pdc["n_frequency_bins"]),
        spectral_range_hz=pdc["spectral_range_hz"],
        roi_channels=pdc["roi_channels"],
        bands_hz=pdc["bands_hz"],
        thresholds=thresholds,
        summary_metric=str(pdc.get("summary_metric", PAPER_PDC_METRIC)),
        n_jobs=n_jobs,
    )


def _run_regression(
    config_path: Path,
    config: Mapping[str, object],
    statistics: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    regression = config.get("pdc_regression", {})
    if not isinstance(regression, Mapping):
        raise ValueError("pdc_regression must be a mapping")
    if not bool(regression.get("enabled", True)):
        _write_json(
            output_dir / "pdc_method_regression.json",
            {
                "status": "skipped",
                "reason": "pdc_regression.enabled=false",
                "failures": [],
            },
        )
        return
    expected_name = regression.get(
        "expected_statistics_file", "pdc_corrected_expected.json"
    )
    expected_path = Path(str(expected_name))
    if not expected_path.is_absolute():
        expected_path = config_path.parent / expected_path
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    tolerance = float(
        regression.get(
            "absolute_tolerance", expected.get("absolute_tolerance", 1e-12)
        )
    )
    report = compare_pdc_statistics(statistics, expected, absolute_tolerance=tolerance)
    report["expected_statistics_file"] = expected_path.name
    _write_json(output_dir / "pdc_method_regression.json", report)
    if report["status"] == "fail":
        raise ValueError(f"PDC method regression failed: {'; '.join(report['failures'][:5])}")


def run_pdc(
    config_path: Path,
    output_dir: Path,
    *,
    n_jobs: int,
    resume: bool,
) -> int:
    """Run QC, null simulation, corrected PDC, statistics, figures, and provenance."""

    config = _read_config(config_path)
    pdc, neuro, paths = config["pdc"], config["neuro"], config["paths"]
    if not all(isinstance(item, Mapping) for item in (pdc, neuro, paths)):
        raise ValueError("pdc, neuro, and paths must be mappings")
    _validate_pdc_config(pdc)
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    visual = str(neuro.get("visual_condition", "st_no"))
    tactile = str(neuro.get("tactile_condition", "st_tf2"))
    metric = str(pdc.get("summary_metric", PAPER_PDC_METRIC))
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = _build_registry(config)
    input_report, header_report = _quality_reports(config, registry)
    _write_quality_outputs(output_dir, registry, input_report, header_report)
    _raise_for_quality(input_report, header_report)
    thresholds, threshold_metadata = _load_or_compute_thresholds(output_dir, pdc, n_jobs=n_jobs)

    runtime_config = {
        "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "pdc": pdc,
        "subject_condition_event_codes": neuro["subject_condition_event_codes"],
        "threshold_settings_sha256": threshold_metadata["settings_sha256"],
    }
    config_sha256 = hashlib.sha256(
        json.dumps(runtime_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint_path = output_dir / "pdc_checkpoint.json"
    results: list[dict[str, object]] = []
    if resume and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("config_sha256") != config_sha256:
            raise ValueError("Checkpoint config hash does not match the requested config")
        results = list(checkpoint.get("results", []))
        if any("pdc_by_direction_band" not in result for result in results):
            raise ValueError("Checkpoint was produced by an incompatible PDC schema")
    completed = {(result["subject"], result["condition"]) for result in results}
    for index, item in enumerate(registry, start=1):
        key = (item.subject, item.condition)
        if key in completed:
            print(
                f"PDC {index}/{len(registry)}: "
                f"{item.subject}/{item.condition} cached",
                flush=True,
            )
            continue
        print(f"PDC {index}/{len(registry)}: {item.subject}/{item.condition}", flush=True)
        results.append(_compute_item(item, pdc, thresholds, n_jobs=n_jobs))
        _write_json(
            checkpoint_path,
            {"status": "in_progress", "config_sha256": config_sha256, "results": results},
        )

    summary = paper_pdc_results_to_frame(results, metric_name=metric)
    statistics = summarize_paper_pdc(
        summary, visual_condition=visual, tactile_condition=tactile
    )
    summary.to_csv(output_dir / "pdc_subject_summary.csv", index=False)
    pd.DataFrame(statistics).to_csv(output_dir / "pdc_statistics.csv", index=False)
    _write_json(
        output_dir / "pdc_statistics.json",
        {
            "schema_version": 1,
            "analysis_status": "corrected_explicit_fixed_order_method",
            "metric": metric,
            "metric_definition": PDC_METRIC_DEFINITIONS[metric],
            "difference_definition": f"{visual} minus {tactile}",
            "statistics": statistics,
        },
    )
    _run_regression(config_path, config, statistics, output_dir)

    from .pdc_figures import plot_pdc_behavior_correlation, plot_pdc_summary

    plot_pdc_summary(
        summary,
        statistics,
        visual_condition=visual,
        tactile_condition=tactile,
        output_paths=(
            output_dir / "figure_7_pdc_corrected.png",
            output_dir / "figure_7_pdc_corrected.pdf",
        ),
    )
    outputs = [
        "pdc_input_quality.json",
        "pdc_header_quality.json",
        "pdc_input_registry.csv",
        "pdc_null_threshold.npz",
        "pdc_null_threshold.json",
        "pdc_subject_summary.csv",
        "pdc_statistics.csv",
        "pdc_statistics.json",
        "pdc_method_regression.json",
        "figure_7_pdc_corrected.png",
        "figure_7_pdc_corrected.pdf",
    ]
    behavior_name = paths.get("behavior_subject_summary")
    if behavior_name:
        behavior = pd.read_csv(Path(str(behavior_name)))
        correlations = correlate_pdc_with_behavior(summary, behavior)
        pd.DataFrame(correlations).to_csv(output_dir / "pdc_behavior_correlations.csv", index=False)
        _write_json(output_dir / "pdc_behavior_correlations.json", correlations)
        plot_pdc_behavior_correlation(
            summary,
            behavior,
            correlations,
            behavior_column="force_cv",
            output_paths=(
                output_dir / "figure_8_pdc_behavior_corrected.png",
                output_dir / "figure_8_pdc_behavior_corrected.pdf",
            ),
        )
        outputs.extend(
            [
                "pdc_behavior_correlations.csv",
                "pdc_behavior_correlations.json",
                "figure_8_pdc_behavior_corrected.png",
                "figure_8_pdc_behavior_corrected.pdf",
            ]
        )

    import mne
    import scipy

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "pdc_corrected_explicit_fixed_order_method",
        "pipeline_version": __version__,
        "randomness": {
            "scope": "Monte Carlo null threshold only",
            "seed": int(pdc["monte_carlo_random_seed"]),
        },
        "config": {
            "file": config_path.name,
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "runtime_sha256": config_sha256,
            "pdc": pdc,
            "pdc_regression": config.get("pdc_regression", {}),
        },
        "data": {
            "quality_status": "pass",
            "dataset_count": len(registry),
            "datasets": [
                {
                    "subject": item.subject,
                    "condition": item.condition,
                    "event_code": item.event_code,
                    "eeg_set": item.eeg_set.name,
                    "emg_set": item.emg_set.name,
                    "annotation": item.annotation.name,
                }
                for item in registry
            ],
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "mne": mne.__version__,
            "pdc_implementation": "internal SCoT-compatible Baccala 2001 formula",
        },
        "outputs": outputs,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(
        checkpoint_path,
        {"status": "complete", "config_sha256": config_sha256, "results": results},
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vtf-pdc")
    commands = parser.add_subparsers(dest="command", required=True)
    qc = commands.add_parser("qc", help="Validate configured EEG/EMG inputs")
    qc.add_argument("--config", type=Path, required=True)
    qc.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run", help="Run the corrected explicit PDC method")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--n-jobs", type=int, default=1)
    run.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "qc":
        return run_pdc_qc(args.config, args.output_dir)
    if args.command == "run":
        return run_pdc(args.config, args.output_dir, n_jobs=args.n_jobs, resume=args.resume)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

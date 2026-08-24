"""Command-line entry point for the corrected CMC analysis."""

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
from .cmc import PAPER_CMC_METRIC, PAPER_CMC_METRIC_DEFINITION
from .cmc_batch import paper_cmc_results_to_frame, summarize_paper_cmc
from .cmc_pipeline import compute_cmc_trace
from .cmc_regression import compare_cmc_statistics
from .neuro_registry import (
    NeuroInput,
    build_neuro_registry,
    profile_neuro_headers,
    profile_neuro_inputs,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_config(config_path: Path) -> dict[str, object]:
    with config_path.open("rb") as stream:
        return tomllib.load(stream)


def _validate_cmc_config(cmc: Mapping[str, object]) -> None:
    if cmc.get("emg_filter_after_alignment") != "none":
        raise ValueError("CMC requires no additional EMG filtering after alignment")
    if bool(cmc.get("emg_rectification")):
        raise ValueError("CMC requires unrectified EMG")
    if cmc.get("estimator") != "multitaper magnitude-squared coherence":
        raise ValueError("CMC estimator must be multitaper magnitude-squared coherence")
    if not bool(cmc.get("apply_csd")):
        raise ValueError("CMC paper method requires current-source density")
    metric = str(cmc.get("summary_metric", PAPER_CMC_METRIC))
    if metric != PAPER_CMC_METRIC:
        raise ValueError(
            f"Stable CMC summary metric must be {PAPER_CMC_METRIC!r}; got {metric!r}"
        )

    trial_indices = list(cmc["trial_indices"])
    analysis_duration = float(cmc["tmax_s"]) - float(cmc["tmin_s"])
    segment_duration = float(cmc["segment_duration_s"])
    calculated = len(trial_indices) * analysis_duration / segment_duration
    rounded = int(round(calculated))
    if rounded <= 0 or abs(calculated - rounded) > 1e-9:
        raise ValueError("CMC analysis window must contain a whole number of segments")
    if int(cmc.get("expected_segment_count", rounded)) != rounded:
        raise ValueError(
            "expected_segment_count does not match trial/window segmentation"
        )


def _build_registry(config: Mapping[str, object]) -> list[NeuroInput]:
    paths = config["paths"]
    neuro = config["neuro"]
    if not isinstance(paths, Mapping) or not isinstance(neuro, Mapping):
        raise ValueError("paths and neuro must be configuration mappings")
    return build_neuro_registry(
        set_dir=paths["eeglab_set_dir"],
        annotation_dir=paths["annotation_dir"],
        subjects=neuro["subjects"],
        condition_event_codes=neuro["subject_condition_event_codes"],
        epoch_bounds_label=str(neuro.get("epoch_bounds_label", "[0 60]")),
    )


def _quality_reports(
    config: Mapping[str, object],
    registry: Sequence[NeuroInput],
) -> tuple[dict[str, object], dict[str, object]]:
    neuro = config["neuro"]
    cmc = config["cmc"]
    if not isinstance(neuro, Mapping) or not isinstance(cmc, Mapping):
        raise ValueError("neuro and cmc must be configuration mappings")
    input_report = profile_neuro_inputs(
        registry,
        start_annotation=str(neuro.get("start_annotation", "DC trigger 13")),
        stop_annotation=str(neuro.get("stop_annotation", "DC trigger 14")),
        expected_epoch_count=int(neuro.get("expected_epoch_count", 5)),
        expected_trial_duration_s=float(
            neuro.get("expected_trial_duration_s", 60.0)
        ),
        trial_duration_tolerance_s=float(
            neuro.get("trial_duration_tolerance_s", 0.1)
        ),
    )
    header_report = profile_neuro_headers(
        registry,
        expected_epoch_count=int(neuro.get("expected_epoch_count", 5)),
        expected_sampling_rate_hz=float(cmc["expected_sampling_rate_hz"]),
        expected_sample_count=int(
            round(float(cmc["tmax_s"]) * float(cmc["expected_sampling_rate_hz"]))
        ),
        expected_eeg_channel_count=int(neuro.get("expected_eeg_channel_count", 61)),
        expected_emg_channel_count=int(neuro.get("expected_emg_channel_count", 1)),
        required_eeg_channels=cmc["roi_channels"],
    )
    return input_report, header_report


def _raise_for_quality(
    input_report: Mapping[str, object], header_report: Mapping[str, object]
) -> None:
    issues = list(input_report.get("issues", [])) + list(
        header_report.get("issues", [])
    )
    if input_report.get("status") != "pass" or header_report.get("status") != "pass":
        preview = "; ".join(str(issue) for issue in issues[:5])
        raise ValueError(f"CMC input quality checks failed: {preview}")


def _write_quality_outputs(
    output_dir: Path,
    registry: Sequence[NeuroInput],
    input_report: Mapping[str, object],
    header_report: Mapping[str, object],
) -> None:
    _write_json(output_dir / "cmc_input_quality.json", input_report)
    _write_json(output_dir / "cmc_header_quality.json", header_report)
    pd.DataFrame(item.to_record() for item in registry).to_csv(
        output_dir / "cmc_input_registry.csv", index=False
    )


def run_cmc_qc(config_path: Path, output_dir: Path) -> int:
    """Validate all configured EEG/EMG inputs without loading FDT signals."""

    config = _read_config(config_path)
    cmc = config["cmc"]
    if not isinstance(cmc, Mapping):
        raise ValueError("cmc must be a configuration mapping")
    _validate_cmc_config(cmc)
    registry = _build_registry(config)
    input_report, header_report = _quality_reports(config, registry)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_quality_outputs(
        output_dir, registry, input_report, header_report
    )
    _raise_for_quality(input_report, header_report)
    return 0


def _compute_item(
    item: NeuroInput,
    cmc: Mapping[str, object],
    *,
    n_jobs: int,
) -> dict[str, object]:
    return compute_cmc_trace(
        item,
        trial_indices=cmc["trial_indices"],
        tmin_s=float(cmc["tmin_s"]),
        tmax_s=float(cmc["tmax_s"]),
        segment_duration_s=float(cmc["segment_duration_s"]),
        apply_csd=bool(cmc["apply_csd"]),
        csd_stiffness=float(cmc["csd_stiffness"]),
        csd_n_legendre_terms=int(cmc["csd_n_legendre_terms"]),
        csd_lambda2=float(cmc["csd_lambda2"]),
        spectral_range_hz=cmc["spectral_range_hz"],
        multitaper_bandwidth_hz=float(cmc["multitaper_bandwidth_hz"]),
        multitaper_adaptive=bool(cmc["multitaper_adaptive"]),
        roi_channels=cmc["roi_channels"],
        bands_hz=cmc["bands_hz"],
        confidence_alpha=float(cmc["confidence_alpha"]),
        summary_metric=str(cmc.get("summary_metric", PAPER_CMC_METRIC)),
        n_jobs=n_jobs,
    )


def _expected_baseline(
    config_path: Path, config: Mapping[str, object]
) -> tuple[Path, dict[str, object], float]:
    regression = config.get("cmc_regression", {})
    if not isinstance(regression, Mapping):
        raise ValueError("cmc_regression must be a configuration mapping")
    expected_name = regression.get("expected_statistics_file")
    if not expected_name:
        raise ValueError("cmc_regression.expected_statistics_file is required")
    expected_path = Path(str(expected_name))
    if not expected_path.is_absolute():
        expected_path = config_path.parent / expected_path
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    tolerance = float(
        regression.get(
            "absolute_tolerance", expected.get("absolute_tolerance", 1e-12)
        )
    )
    return expected_path, expected, tolerance


def run_cmc(
    config_path: Path,
    output_dir: Path,
    *,
    n_jobs: int,
    resume: bool,
) -> int:
    """Run QC, corrected CMC, regression, plotting, and provenance."""

    config = _read_config(config_path)
    cmc = config["cmc"]
    neuro = config["neuro"]
    if not isinstance(cmc, Mapping) or not isinstance(neuro, Mapping):
        raise ValueError("cmc and neuro must be configuration mappings")
    _validate_cmc_config(cmc)
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    visual = str(neuro.get("visual_condition", "st_no"))
    tactile = str(neuro.get("tactile_condition", "st_tf2"))
    metric = str(cmc.get("summary_metric", PAPER_CMC_METRIC))

    output_dir.mkdir(parents=True, exist_ok=True)
    registry = _build_registry(config)
    input_report, header_report = _quality_reports(config, registry)
    _write_quality_outputs(
        output_dir, registry, input_report, header_report
    )
    _raise_for_quality(input_report, header_report)

    runtime_config = {
        "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "cmc": cmc,
        "subject_condition_event_codes": neuro["subject_condition_event_codes"],
    }
    config_sha256 = hashlib.sha256(
        json.dumps(runtime_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint_path = output_dir / "cmc_checkpoint.json"
    results: list[dict[str, object]] = []
    if resume and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("config_sha256") != config_sha256:
            raise ValueError("Checkpoint config hash does not match the requested config")
        results = list(checkpoint.get("results", []))
        if any("cmc_by_band" not in result for result in results):
            raise ValueError("Checkpoint was produced by an incompatible CMC schema")
    completed = {(result["subject"], result["condition"]) for result in results}

    for index, item in enumerate(registry, start=1):
        key = (item.subject, item.condition)
        if key in completed:
            print(
                f"CMC {index}/{len(registry)}: {item.subject}/{item.condition} cached",
                flush=True,
            )
            continue
        print(
            f"CMC {index}/{len(registry)}: {item.subject}/{item.condition}",
            flush=True,
        )
        results.append(_compute_item(item, cmc, n_jobs=n_jobs))
        _write_json(
            checkpoint_path,
            {
                "status": "in_progress",
                "config_sha256": config_sha256,
                "results": results,
            },
        )

    summary = paper_cmc_results_to_frame(results, metric_name=metric)
    statistics = summarize_paper_cmc(
        summary,
        visual_condition=visual,
        tactile_condition=tactile,
    )
    summary.to_csv(output_dir / "cmc_subject_summary.csv", index=False)
    pd.DataFrame(statistics).to_csv(output_dir / "cmc_statistics.csv", index=False)
    _write_json(
        output_dir / "cmc_statistics.json",
        {
            "schema_version": 1,
            "analysis_status": "corrected_single_run_method",
            "metric": metric,
            "metric_definition": PAPER_CMC_METRIC_DEFINITION,
            "difference_definition": f"{visual} minus {tactile}",
            "statistics": statistics,
        },
    )

    expected_path, expected, tolerance = _expected_baseline(config_path, config)
    if expected.get("metric") != metric:
        raise ValueError("Configured CMC metric does not match regression baseline")
    regression_report = compare_cmc_statistics(
        statistics,
        expected,
        absolute_tolerance=tolerance,
    )
    regression_report["expected_statistics_file"] = expected_path.name
    _write_json(output_dir / "cmc_method_regression.json", regression_report)
    if regression_report["status"] == "fail":
        preview = "; ".join(regression_report["failures"][:5])
        raise ValueError(f"CMC method regression failed: {preview}")

    from .cmc_figures import plot_cmc_summary

    plot_cmc_summary(
        summary,
        statistics,
        visual_condition=visual,
        tactile_condition=tactile,
        output_paths=(
            output_dir / "figure_5_cmc_corrected.png",
            output_dir / "figure_5_cmc_corrected.pdf",
        ),
    )

    import mne
    import mne_connectivity
    import numpy as np
    import scipy

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "cmc_corrected_single_run_method",
        "pipeline_version": __version__,
        "randomness": "none",
        "config": {
            "file": config_path.name,
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "runtime_sha256": config_sha256,
            "cmc": cmc,
            "cmc_regression": config.get("cmc_regression", {}),
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
            "mne_connectivity": mne_connectivity.__version__,
        },
        "outputs": [
            "cmc_input_quality.json",
            "cmc_header_quality.json",
            "cmc_input_registry.csv",
            "cmc_subject_summary.csv",
            "cmc_statistics.csv",
            "cmc_statistics.json",
            "cmc_method_regression.json",
            "figure_5_cmc_corrected.png",
            "figure_5_cmc_corrected.pdf",
        ],
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(
        checkpoint_path,
        {
            "status": "complete",
            "config_sha256": config_sha256,
            "results": results,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vtf-cmc")
    commands = parser.add_subparsers(dest="command", required=True)
    qc = commands.add_parser("qc", help="Validate the configured EEG/EMG inputs.")
    qc.add_argument("--config", type=Path, required=True)
    qc.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run", help="Run the corrected single-run CMC method.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--n-jobs", type=int, default=1)
    run.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "qc":
        return run_cmc_qc(args.config, args.output_dir)
    if args.command == "run":
        return run_cmc(
            args.config,
            args.output_dir,
            n_jobs=args.n_jobs,
            resume=args.resume,
        )
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point for the paper behavior analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from . import __version__
from .behavior import analyze_force_dataset
from .quality import profile_behavior_dataset
from .regression import compare_behavior_statistics
from .statistics import paired_wilcoxon_signed_rank


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_behavior(config_path: Path, output_dir: Path) -> int:
    """Run QC, behavior metrics, paper regression, plotting, and provenance."""

    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    paths = config["paths"]
    behavior = config["behavior"]
    conditions = behavior["conditions"]
    offset = float(behavior["analysis_start_offset_ms"])
    start_offsets = {condition: offset for condition in conditions}
    file_template = behavior.get("file_template", "{subject}_{condition}.csv")
    subject_subdirectories = behavior.get("subject_subdirectories")

    output_dir.mkdir(parents=True, exist_ok=True)
    quality_config = config.get("quality", {})
    quality_report = profile_behavior_dataset(
        data_dir=paths["force_data_dir"],
        subjects=behavior["subjects"],
        conditions=conditions,
        trial_indices=behavior["trial_indices"],
        start_offsets_ms=start_offsets,
        file_template=file_template,
        subject_subdirectories=subject_subdirectories,
        expected_trial_count=quality_config.get("expected_trial_count", 5),
        expected_sampling_rate_hz=quality_config.get(
            "expected_sampling_rate_hz", 500.0
        ),
        sampling_rate_tolerance_hz=float(
            quality_config.get("sampling_rate_tolerance_hz", 1e-6)
        ),
        compute_sha256=bool(quality_config.get("compute_sha256", False)),
    )
    _write_json(output_dir / "behavior_data_quality.json", quality_report)
    if quality_report["status"] != "pass":
        preview = "; ".join(str(item) for item in quality_report["issues"][:5])
        raise ValueError(f"Behavior data quality checks failed: {preview}")

    trial_table, summary_table = analyze_force_dataset(
        data_dir=paths["force_data_dir"],
        subjects=behavior["subjects"],
        conditions=conditions,
        trial_indices=behavior["trial_indices"],
        start_offsets_ms=start_offsets,
        file_template=file_template,
        subject_subdirectories=subject_subdirectories,
        lowpass_hz=float(behavior["lowpass_hz"]),
        filter_order=int(behavior["filter_order"]),
        normalization=behavior["normalization"],
        ddof=int(behavior["standard_deviation_ddof"]),
    )

    visual = behavior["visual_condition"]
    tactile = behavior["tactile_condition"]
    wide_mean = summary_table.pivot(
        index="subject", columns="condition", values="mean_force"
    )
    wide_cv = summary_table.pivot(
        index="subject", columns="condition", values="force_cv"
    )
    statistics = {
        "difference_definition": f"{visual} minus {tactile}",
        "mean_force": paired_wilcoxon_signed_rank(
            wide_mean[visual].to_numpy(),
            wide_mean[tactile].to_numpy(),
        ).to_dict(),
        "force_cv": paired_wilcoxon_signed_rank(
            wide_cv[visual].to_numpy(),
            wide_cv[tactile].to_numpy(),
        ).to_dict(),
    }

    trial_table.to_csv(output_dir / "behavior_trials.csv", index=False)
    summary_table.to_csv(output_dir / "behavior_subject_summary.csv", index=False)
    _write_json(output_dir / "behavior_statistics.json", statistics)

    regression_config = config.get("regression", {})
    expected_file_name = regression_config.get("expected_statistics_file")
    if expected_file_name:
        expected_path = Path(expected_file_name)
        if not expected_path.is_absolute():
            expected_path = config_path.parent / expected_path
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        tolerance = float(
            regression_config.get(
                "absolute_tolerance",
                expected.get("absolute_tolerance", 1e-12),
            )
        )
        regression_report = compare_behavior_statistics(
            statistics,
            expected,
            absolute_tolerance=tolerance,
        )
        regression_report["expected_statistics_file"] = expected_path.name
    else:
        regression_report = {"status": "not_requested", "failures": []}
    _write_json(output_dir / "behavior_paper_regression.json", regression_report)
    if regression_report["status"] == "fail":
        preview = "; ".join(str(item) for item in regression_report["failures"][:5])
        raise ValueError(f"Behavior paper regression failed: {preview}")

    from .figures import plot_behavior_figure

    plot_behavior_figure(
        summary_table=summary_table,
        statistics=statistics,
        visual_condition=visual,
        tactile_condition=tactile,
        output_paths=(
            output_dir / "figure_3_behavior.png",
            output_dir / "figure_3_behavior.pdf",
        ),
    )

    import matplotlib
    import numpy as np
    import pandas as pd
    import scipy

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "behavior",
        "pipeline_version": __version__,
        "randomness": "none",
        "config": {
            "file": config_path.name,
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "behavior": behavior,
            "quality": quality_config,
            "regression": regression_config,
        },
        "data": {
            "root": str(Path(paths["force_data_dir"])),
            "quality_status": quality_report["status"],
            "files": [
                {
                    key: file_report[key]
                    for key in ("file", "size_bytes", "sha256")
                    if key in file_report
                }
                for file_report in quality_report["files"]
                if file_report["present"]
            ],
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": [
            "behavior_data_quality.json",
            "behavior_trials.csv",
            "behavior_subject_summary.csv",
            "behavior_statistics.json",
            "behavior_paper_regression.json",
            "figure_3_behavior.png",
            "figure_3_behavior.pdf",
        ],
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vtf-behavior")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the paper behavior analysis.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_behavior(args.config, args.output_dir)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

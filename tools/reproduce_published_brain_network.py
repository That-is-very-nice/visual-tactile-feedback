"""Reproduce the archived paper brain-network max-T edge set from local paths."""

from __future__ import annotations

import argparse
from pathlib import Path

from visual_tactile_force.brain_network_cli import run_legacy_regression


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    return run_legacy_regression(arguments.config, arguments.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())

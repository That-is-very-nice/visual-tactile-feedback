"""Verify saved brain-network Wilcoxon-Holm results."""

from __future__ import annotations

import argparse
from pathlib import Path

from visual_tactile_force.brain_network_cli import run_verify_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    return run_verify_results(arguments.config, arguments.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())

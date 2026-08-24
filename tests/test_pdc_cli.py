from __future__ import annotations

import unittest
from pathlib import Path

from visual_tactile_force.pdc_cli import (
    _read_config,
    _validate_pdc_config,
    build_parser,
)


class PdcCliTests(unittest.TestCase):
    def test_example_configuration_passes(self) -> None:
        config = _read_config(Path("configs/pdc.example.toml"))
        _validate_pdc_config(config["pdc"])

    def test_scalar_null_threshold_is_rejected(self) -> None:
        config = _read_config(Path("configs/pdc.example.toml"))
        config["pdc"]["null_threshold_type"] = "scalar"
        with self.assertRaisesRegex(ValueError, "frequency_resolved"):
            _validate_pdc_config(config["pdc"])

    def test_run_parser_exposes_resume(self) -> None:
        arguments = build_parser().parse_args(
            [
                "run",
                "--config",
                "configs/pdc.example.toml",
                "--output-dir",
                "output",
                "--resume",
            ]
        )
        self.assertEqual(arguments.command, "run")
        self.assertTrue(arguments.resume)


if __name__ == "__main__":
    unittest.main()

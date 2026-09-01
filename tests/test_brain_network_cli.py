from __future__ import annotations

import unittest
from pathlib import Path

from visual_tactile_force.brain_network_cli import (
    _read_config,
    _validate_network_config,
    build_parser,
)


class BrainNetworkCliTests(unittest.TestCase):
    def test_example_configuration_passes(self) -> None:
        config = _read_config(Path("configs/brain_network.example.toml"))
        _validate_network_config(config["brain_network"])
        self.assertTrue(config["brain_network"]["include_within_roi"])
        self.assertEqual(config["brain_network"]["statistics"]["family_size"], 100)

    def test_midline_channel_cannot_be_added_to_frozen_map(self) -> None:
        config = _read_config(Path("configs/brain_network.example.toml"))
        config["brain_network"]["roi_channels"]["Left_Central"].append("Cz")
        with self.assertRaisesRegex(ValueError, "frozen paper ROI map"):
            _validate_network_config(config["brain_network"])

    def test_run_parser_exposes_resume(self) -> None:
        arguments = build_parser().parse_args(
            [
                "run",
                "--config",
                "configs/brain_network.example.toml",
                "--output-dir",
                "output",
                "--resume",
            ]
        )
        self.assertEqual(arguments.command, "run")
        self.assertTrue(arguments.resume)

    def test_verify_parser_is_available(self) -> None:
        arguments = build_parser().parse_args(
            [
                "verify",
                "--config",
                "configs/brain_network.example.toml",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(arguments.command, "verify")


if __name__ == "__main__":
    unittest.main()

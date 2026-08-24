from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from visual_tactile_force.neuro_registry import (
    build_neuro_registry,
    profile_neuro_inputs,
)


class NeuroRegistryTests(unittest.TestCase):
    def test_complete_pair_and_five_epochs_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            set_dir = root / "set"
            annotation_dir = root / "annotation"
            set_dir.mkdir()
            annotation_dir.mkdir()
            stem = "s11_[0 60]"
            for suffix in ("EEG.set", "EEG.fdt", "EMG.set", "EMG.fdt"):
                (set_dir / f"{stem}_{suffix}").write_bytes(b"fixture")
            rows = ["Onset,Annotation"]
            rows.append("-1.0,DC trigger 14")
            for trial in range(5):
                rows.extend(
                    [
                        f"+{trial * 100}.0,DC trigger 13",
                        f"+{trial * 100 + 60}.0,DC trigger 14",
                    ]
                )
            (annotation_dir / "eeg_s11_61_5_annotations.txt").write_text(
                "\n".join(rows) + "\n", encoding="utf-8"
            )

            registry = build_neuro_registry(
                set_dir=set_dir,
                annotation_dir=annotation_dir,
                subjects=["s1"],
                condition_event_codes={"st_no": 1},
            )
            report = profile_neuro_inputs(registry)

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["passed_dataset_count"], 1)
            self.assertEqual(report["datasets"][0]["annotation_counts"]["DC trigger 13"], 5)
            self.assertEqual(report["datasets"][0]["ignored_pretrial_stop_count"], 1)

    def test_missing_companion_fdt_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            registry = build_neuro_registry(
                set_dir=root,
                annotation_dir=root,
                subjects=["s1"],
                condition_event_codes={"st_no": 1},
            )
            report = profile_neuro_inputs(registry)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("Missing eeg_fdt" in issue for issue in report["issues"]))

    def test_subject_specific_event_codes_are_not_flattened(self) -> None:
        registry = build_neuro_registry(
            set_dir="/data/set",
            annotation_dir="/data/annotation",
            subjects=["qh", "wxl"],
            condition_event_codes={
                "qh": {"st_no": 1, "st_tf2": 8},
                "wxl": {"st_no": 1, "st_tf2": 9},
            },
        )
        codes = {
            (item.subject, item.condition): item.event_code for item in registry
        }
        self.assertEqual(codes[("qh", "st_tf2")], 8)
        self.assertEqual(codes[("wxl", "st_tf2")], 9)


if __name__ == "__main__":
    unittest.main()

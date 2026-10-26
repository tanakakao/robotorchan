"""Structural tests for the fixed theory-guide chapter contract."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_theory_guide_quality.py"
SPEC = importlib.util.spec_from_file_location("check_theory_guide_quality", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GuideQualityTests(unittest.TestCase):
    def test_fixed_chapter_counts(self) -> None:
        self.assertEqual(len(MODULE.MAIN_CHAPTERS), 24)
        self.assertEqual(len(MODULE.ACQUISITION_CHAPTERS), 12)
        self.assertEqual(len(MODULE.OPTIMIZATION_CHAPTERS), 12)
        self.assertEqual(len(set(MODULE.expected_chapters())), 48)

    def test_current_repository(self) -> None:
        self.assertEqual(MODULE.check_guide_quality(), [])

    def test_missing_docs_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = MODULE.check_guide_quality(Path(directory))
        self.assertIn("missing original mapping: models.md", failures)
        self.assertIn("missing guide chapter: 01_bayesian_optimization.md", failures)

    def test_broken_chapter_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in MODULE.MAPPING_ORIGINALS:
                dest = root / MODULE.MAPPING / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("# preserved\n", encoding="utf-8")
            for relative in MODULE.expected_chapters():
                src = root / MODULE.THEORY / relative
                dst = root / MODULE.MAPPING / "guide" / relative
                src.parent.mkdir(parents=True, exist_ok=True)
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.write_text("# theory\n", encoding="utf-8")
                dst.write_text("# Guide\n\n## Explanation\n\n[Theory](../../../theory/ref.md)\n", encoding="utf-8")
            broken = root / MODULE.MAPPING / "guide" / "13_model_selection.md"
            broken.write_text("# only a heading\n", encoding="utf-8")
            failures = MODULE.check_guide_quality(root)
        self.assertIn("missing explanatory sections: 13_model_selection.md", failures)
        self.assertIn("missing theory navigation: 13_model_selection.md", failures)


if __name__ == "__main__":
    unittest.main()

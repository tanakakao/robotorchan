"""Regression tests for the generated model documentation coverage contract."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from robotorchan.models.registry import MODEL_REGISTRY
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_model_coverage.py"
spec = importlib.util.spec_from_file_location("generate_model_coverage", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load model coverage generator: {SCRIPT_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
build_model_coverage = module.build_model_coverage
render_model_coverage = module.render_model_coverage


class ModelCoverageTests(unittest.TestCase):
    def test_coverage_tracks_registry_entries(self) -> None:
        coverage = build_model_coverage()
        self.assertEqual(set(coverage["models"]), set(MODEL_REGISTRY))
        self.assertEqual(coverage["source_of_truth"], "robotorchan.models.registry.MODEL_REGISTRY")

    def test_coverage_entries_have_expected_fields(self) -> None:
        coverage = build_model_coverage()
        for model in coverage["models"].values():
            with self.subTest(model=model):
                self.assertEqual(set(model), {"guide", "theory", "notebook"})

    def test_generated_artifact_is_deterministic(self) -> None:
        self.assertEqual(render_model_coverage(), render_model_coverage())
        self.assertTrue(render_model_coverage().endswith("\n"))


if __name__ == "__main__":
    unittest.main()

"""Tests for the theory–implementation navigation consistency checker."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_theory_implementation_mapping.py"
SPEC = importlib.util.spec_from_file_location("check_theory_implementation_mapping", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MappingConsistencyTests(unittest.TestCase):
    def test_current_repository_navigation(self) -> None:
        self.assertEqual(MODULE.check_mapping_consistency(), [])

    def test_missing_document_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = MODULE.check_mapping_consistency(Path(directory))
        self.assertIn("missing mapping document: README.md", failures)

    def test_missing_return_link_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "docs" / "development" / "theory-implementation"
            mapping.mkdir(parents=True)
            for name in MODULE.FILES:
                (mapping / name).write_text("", encoding="utf-8")
            failures = MODULE.check_mapping_consistency(root)
        self.assertIn("models.md missing navigation link: verification.md", failures)


if __name__ == "__main__":
    unittest.main()

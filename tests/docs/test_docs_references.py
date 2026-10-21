"""Regression tests for Markdown reference links and heading fragments."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "check_docs_references", SCRIPTS / "check_docs_references.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DocumentationReferenceTests(unittest.TestCase):
    def test_duplicate_heading_anchors(self) -> None:
        self.assertEqual(MODULE.anchors("# Same\n## Same\n"), {"same", "same-1"})

    def test_reference_style_link_to_existing_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "target.md").write_text("# Section\n", encoding="utf-8")
            (docs / "README.md").write_text(
                "[read][target]\n\n[target]: target.md#section\n",
                encoding="utf-8",
            )
            self.assertEqual(MODULE.check_reference_anchors(root), [])

    def test_missing_reference_definition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "README.md").write_text("[read][missing]\n", encoding="utf-8")
            self.assertEqual(
                MODULE.check_reference_anchors(root),
                ["docs/README.md: undefined reference: missing"],
            )

    def test_missing_heading_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "target.md").write_text("# Existing\n", encoding="utf-8")
            (docs / "README.md").write_text(
                "[read](target.md#absent)\n", encoding="utf-8"
            )
            self.assertEqual(
                MODULE.check_reference_anchors(root),
                ["docs/README.md: missing anchor: target.md#absent"],
            )


if __name__ == "__main__":
    unittest.main()

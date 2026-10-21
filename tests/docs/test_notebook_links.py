"""Regression tests for notebook Markdown link validation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "check_notebook_links", SCRIPTS / "check_notebook_links.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NotebookLinkTests(unittest.TestCase):
    def _write_notebook(self, root: Path, cells: list[dict]) -> None:
        notebooks = root / "examples" / "notebooks"
        notebooks.mkdir(parents=True)
        (notebooks / "example.ipynb").write_text(json.dumps({"cells": cells}), encoding="utf-8")

    def test_valid_markdown_link_and_ignored_code_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
            self._write_notebook(
                root,
                [
                    {"cell_type": "markdown", "source": ["[guide](../../docs/guide.md)"]},
                    {"cell_type": "code", "source": ["[bad](missing.md)"]},
                ],
            )
            self.assertEqual(MODULE.check_notebook_links(root), [])

    def test_missing_markdown_target_reports_cell_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_notebook(
                root,
                [
                    {"cell_type": "code", "source": ["pass"]},
                    {"cell_type": "markdown", "source": ["[bad](missing.md)"]},
                ],
            )
            self.assertEqual(
                MODULE.check_notebook_links(root),
                ["examples/notebooks/example.ipynb cell 1: missing target: missing.md"],
            )

    def test_outside_repository_target_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_notebook(
                root,
                [{"cell_type": "markdown", "source": ["[bad](../../../escape.md)"]}],
            )
            expected = "examples/notebooks/example.ipynb cell 0: outside repository: "
            expected += "../../../escape.md"
            self.assertEqual(MODULE.check_notebook_links(root), [expected])


if __name__ == "__main__":
    unittest.main()

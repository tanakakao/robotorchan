"""Regression tests for local documentation link validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_docs_links import check_links, strip_fenced_code


class DocumentationLinkTests(unittest.TestCase):
    def test_valid_file_and_directory_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "target.md").write_text("# Target\n", encoding="utf-8")
            (docs / "guide").mkdir()
            (docs / "README.md").write_text(
                "[file](target.md) [directory](guide/)\n",
                encoding="utf-8",
            )
            self.assertEqual(check_links(root), [])

    def test_missing_link_reports_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "README.md").write_text("[missing](absent.md)\n", encoding="utf-8")
            self.assertEqual(
                check_links(root),
                ["docs/README.md: missing target: absent.md"],
            )

    def test_fenced_code_links_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "README.md").write_text(
                "```md\n[example](absent.md)\n```\n",
                encoding="utf-8",
            )
            self.assertEqual(check_links(root), [])

    def test_fence_stripping_preserves_regular_links(self) -> None:
        text = "[keep](present.md)\n~~~md\n[skip](missing.md)\n~~~\n"
        self.assertIn("[keep](present.md)", strip_fenced_code(text))
        self.assertNotIn("[skip](missing.md)", strip_fenced_code(text))


if __name__ == "__main__":
    unittest.main()

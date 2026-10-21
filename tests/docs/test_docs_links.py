"""Regression tests for local documentation link validation."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_docs_links.py"
SPEC = importlib.util.spec_from_file_location("check_docs_links", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
check_links = MODULE.check_links
strip_fenced_code = MODULE.strip_fenced_code


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

    def test_encoded_local_target_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "target file.md").write_text("# Target\n", encoding="utf-8")
            (docs / "README.md").write_text(
                "[target](target%20file.md)\n",
                encoding="utf-8",
            )
            self.assertEqual(check_links(root), [])

    def test_parent_directory_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "README.md").write_text(
                "[outside](../../outside.md)\n",
                encoding="utf-8",
            )
            self.assertEqual(
                check_links(root),
                ["docs/README.md: outside repository: ../../outside.md"],
            )

    def test_missing_target_with_fragment_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "README.md").write_text(
                "[missing](absent.md#section)\n",
                encoding="utf-8",
            )
            self.assertEqual(
                check_links(root),
                ["docs/README.md: missing target: absent.md#section"],
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

"""Regression checks for migrated engineering and historical documentation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from check_docs_links import check_links
from check_docs_references import check_reference_anchors
from check_documentation_inventory import INVENTORY, check_inventory


class InventoryTests(unittest.TestCase):
    def test_repository_inventory(self) -> None:
        self.assertEqual(check_inventory(), [])

    def test_unclassified_document_and_missing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / INVENTORY
            inventory.parent.mkdir(parents=True)
            inventory.write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "current_path": "old.md",
                                "destination": "missing.md",
                                "category": "Guide",
                                "reason": "Relocation",
                                "action": "MOVE",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "docs").mkdir()
            (root / "docs/new.md").write_text("# New\n", encoding="utf-8")
            failures = check_inventory(root)
        self.assertIn("unclassified Markdown: docs/new.md", failures)
        self.assertIn("missing or unsafe destination: missing.md", failures)

    def test_internal_and_record_links_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("docs-internal", ".github/project-records"):
                path = root / name / "README.md"
                path.parent.mkdir(parents=True)
                path.write_text("# Reference\n\n[Missing](lost.md)\n", encoding="utf-8")
            self.assertEqual(len(check_links(root)), 2)

    def test_reference_without_fragment_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs/index.md").write_text(
                "# Index\n\n[Missing][target]\n\n[target]: missing.md\n", encoding="utf-8"
            )
            failures = check_reference_anchors(root)
        self.assertEqual(len(failures), 1)
        self.assertIn("missing reference target", failures[0])


if __name__ == "__main__":
    unittest.main()

"""Guard the separation of mathematical Theory from implementation guidance."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from check_documentation_boundaries import check_boundaries


class BoundaryTests(unittest.TestCase):
    def test_repository(self) -> None:
        self.assertEqual(check_boundaries(), [])

    def test_implementation_and_broken_source_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            theory = root / "docs/theory/gp.md"
            guide = root / "docs/guide/gp.md"
            source = root / "src/robotorchan/model.py"
            for path in (theory, guide, source):
                path.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("class ExampleGP: pass\n", encoding="utf-8")
            theory.write_text("# Theory\n\nUse ExampleGP from robotorchan.\n", encoding="utf-8")
            guide.write_text("# Guide\n\nsrc/robotorchan/missing.py\nPhase 3\n", encoding="utf-8")
            failures = check_boundaries(root)
        self.assertEqual(len(failures), 4)

    def test_mathematical_method_names_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "docs/theory/decision.md"
            path.parent.mkdir(parents=True)
            path.write_text("# Decision theory\n\nBALD, CVaR and Straddle.\n", encoding="utf-8")
            self.assertEqual(check_boundaries(root), [])


if __name__ == "__main__":
    unittest.main()

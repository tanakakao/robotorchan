"""Execute the remaining structured-output example from the published guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide"


class StructuredModelGuideExamplesTests(unittest.TestCase):
    def run_example(self, chapter: str) -> dict[str, object]:
        path = GUIDE / chapter
        match = re.search(
            r"```python\n(.*?)\n```",
            path.read_text(encoding="utf-8"),
            re.DOTALL,
        )
        self.assertIsNotNone(match, f"Missing Python example in {chapter}")
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(path), "exec"), scope)
        return scope

    def test_higher_order_constructor(self) -> None:
        scope = self.run_example("11_structured_output.md")
        self.assertEqual(scope["X"].shape, (3, 1))
        self.assertEqual(scope["Y"].shape, (3, 2, 2))
        self.assertEqual(type(scope["model"]).__name__, "HigherOrderGP")


if __name__ == "__main__":
    unittest.main()

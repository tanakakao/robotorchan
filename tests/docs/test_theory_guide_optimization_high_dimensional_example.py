"""Execute HeSBO dimensional mapping example directly from the optimization guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/theory-implementation/guide/optimization/09_high_dimensional.md"
)


class HighDimensionalGuideTests(unittest.TestCase):
    def test_hesbo_embedding_and_original_candidates(self) -> None:
        chapter = GUIDE.read_text(encoding="utf-8")
        section = chapter.split("## 6.", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        self.assertEqual(tuple(scope["original"].shape), (1, 6))
        self.assertEqual(tuple(scope["embedded"].shape), (1, 2))
        torch.testing.assert_close(
            scope["strategy"].project(scope["embedded"]), scope["original"]
        )


if __name__ == "__main__":
    unittest.main()

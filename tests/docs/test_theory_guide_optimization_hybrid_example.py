"""Execute the hybrid DE-to-local optimization example in the guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide/optimization/11_hybrid.md"


class HybridOptimizationGuideTests(unittest.TestCase):
    def test_de_to_botorch_local_candidate(self) -> None:
        chapter = GUIDE.read_text(encoding="utf-8")
        section = chapter.split("## 5.", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        self.assertEqual(tuple(scope["candidate"].shape), (1, 1))
        self.assertTrue(torch.isfinite(scope["score"]).all())


if __name__ == "__main__":
    unittest.main()

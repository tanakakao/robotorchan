"""Execute the documented candidate-space linear constraint example."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/theory-implementation/guide/optimization/05_constraints.md"
)


class OptimizationConstraintsGuideTests(unittest.TestCase):
    def test_linear_candidate_feasibility_example(self) -> None:
        chapter = GUIDE.read_text(encoding="utf-8")
        section = chapter.split("## 7.", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        candidate = scope["candidate"]
        self.assertEqual(tuple(candidate.shape), (1, 1))
        self.assertGreaterEqual(candidate[0, 0].item(), 0.3 - 1e-6)
        self.assertLessEqual(candidate[0, 0].item(), 1.0 + 1e-6)
        self.assertTrue(torch.isfinite(scope["value"]).all())


if __name__ == "__main__":
    unittest.main()

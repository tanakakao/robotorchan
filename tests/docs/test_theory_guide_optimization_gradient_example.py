"""Execute the published gradient-based optimization guide example."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/theory-implementation/guide/optimization/02_gradient_based.md"
)


class OptimizationGuideTests(unittest.TestCase):
    def test_gradient_optimizer_dispatch_example(self) -> None:
        chapter = GUIDE.read_text(encoding="utf-8")
        section = chapter.split("## 5.", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        candidate = scope["candidate"]
        bounds = scope["bounds"]
        self.assertEqual(tuple(candidate.shape), (1, 1))
        self.assertTrue(torch.all(candidate >= bounds[0]))
        self.assertTrue(torch.all(candidate <= bounds[1]))


if __name__ == "__main__":
    unittest.main()

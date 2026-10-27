"""Run the native mixed-variable candidate guide example."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/theory-implementation/guide/optimization/04_mixed_discrete.md"
)


class MixedDiscreteGuideTests(unittest.TestCase):
    def test_fixed_feature_enumeration_example(self) -> None:
        chapter = GUIDE.read_text(encoding="utf-8")
        section = chapter.split("## 6.", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        candidate = scope["candidate"]
        self.assertEqual(tuple(candidate.shape), (1, 2))
        self.assertIn(candidate[0, 1].item(), (0.0, 1.0))
        self.assertTrue(torch.all(candidate >= scope["bounds"][0]))
        self.assertTrue(torch.all(candidate <= scope["bounds"][1]))


if __name__ == "__main__":
    unittest.main()

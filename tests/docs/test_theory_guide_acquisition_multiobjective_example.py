"""Execute the two-objective EHVI example from the acquisition guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide/acquisition/07_multiobjective.md"


class MultiobjectiveGuideTests(unittest.TestCase):
    def test_logehvi_two_objective_example(self) -> None:
        chapter = GUIDE.read_text(encoding="utf-8")
        section = chapter.split("## 7.", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        self.assertEqual(tuple(scope["train_Y"].shape), (8, 2))
        self.assertEqual(tuple(scope["ref_point"].shape), (2,))
        self.assertEqual(tuple(scope["score"].shape), (1,))
        self.assertTrue(torch.isfinite(scope["score"]).all())


if __name__ == "__main__":
    unittest.main()

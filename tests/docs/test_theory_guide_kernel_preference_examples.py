"""Run the remaining preference example from the published Japanese guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide"


class KernelPreferenceGuideExamplesTests(unittest.TestCase):
    def run_example(self, chapter: str) -> dict[str, object]:
        path = GUIDE / chapter
        match = re.search(
            r"```python\n(.*?)\n```",
            path.read_text(encoding="utf-8"),
            re.DOTALL,
        )
        self.assertIsNotNone(match, f"Missing Python example: {chapter}")
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(path), "exec"), scope)
        return scope

    def test_pairwise_preference_example(self) -> None:
        scope = self.run_example("10_preference_learning.md")
        model = scope["model"]
        self.assertEqual(tuple(scope["comparisons"].shape), (2, 2))
        self.assertEqual(type(model.make_mll()).__name__, "PairwiseLaplaceMarginalLogLikelihood")
        self.assertEqual(model.num_outputs, 1)


if __name__ == "__main__":
    unittest.main()

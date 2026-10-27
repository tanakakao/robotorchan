"""Execute the published preference-learning example as a regression test."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / (
    "docs/development/theory-implementation/guide/10_preference_learning.md"
)


class PreferenceGuideExampleTests(unittest.TestCase):
    def test_pairwise_example_and_comparison_contract(self) -> None:
        text = GUIDE.read_text(encoding="utf-8")
        match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
        self.assertIsNotNone(match, "Missing Python snippet in preference guide")
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)

        model = scope["model"]
        points = scope["points"]
        comparisons = scope["comparisons"]
        self.assertEqual(tuple(points.shape), (3, 1))
        self.assertEqual(tuple(comparisons.shape), (2, 2))
        self.assertEqual(comparisons.dtype, torch.long)
        self.assertTrue(torch.equal(model.raw_datapoints, points))
        self.assertTrue(torch.equal(model.raw_comparisons, comparisons))
        self.assertEqual(
            type(model.make_mll()).__name__,
            "PairwiseLaplaceMarginalLogLikelihood",
        )


if __name__ == "__main__":
    unittest.main()

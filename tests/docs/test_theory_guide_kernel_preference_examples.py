"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases")


class KernelPreferenceGuideExamplesTests(unittest.TestCase):
    def run_example(self, chapter: str) -> dict[str, object]:
        path = (CASES / chapter).with_suffix(".py")
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        return scope

    def test_pairwise_preference_example(self) -> None:
        scope = self.run_example("10_preference_learning.md")
        model = scope["model"]
        self.assertEqual(tuple(scope["comparisons"].shape), (2, 2))
        self.assertEqual(type(model.make_mll()).__name__, "PairwiseLaplaceMarginalLogLikelihood")
        self.assertEqual(model.num_outputs, 1)


if __name__ == "__main__":
    unittest.main()

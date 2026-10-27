"""Run introductory BO and acquisition examples from the theory guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/development/theory-implementation/guide"


class BayesianOptimizationGuideExamplesTests(unittest.TestCase):
    def run_example(self, chapter: str) -> dict[str, object]:
        path = GUIDE / chapter
        match = re.search(
            r"```python\n(.*?)\n```",
            path.read_text(encoding="utf-8"),
            re.DOTALL,
        )
        self.assertIsNotNone(match, f"Missing Python snippet: {chapter}")
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(path), "exec"), scope)
        return scope

    def test_bo_posterior_example(self) -> None:
        scope = self.run_example("01_bayesian_optimization.md")
        posterior = scope["posterior"]
        self.assertEqual(tuple(posterior.mean.shape), (2, 1))
        self.assertTrue(torch.isfinite(posterior.mean).all())
        self.assertTrue(torch.isfinite(posterior.variance).all())

    def test_acquisition_candidate_example(self) -> None:
        scope = self.run_example("04_acquisition_function.md")
        candidate = scope["candidate"]
        value = scope["acq_value"]
        self.assertEqual(tuple(candidate.shape), (1, 1))
        self.assertTrue(torch.isfinite(candidate).all())
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.all(candidate >= scope["bounds"][0]))
        self.assertTrue(torch.all(candidate <= scope["bounds"][1]))


if __name__ == "__main__":
    unittest.main()

"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "optimization"


class OptimizationSamplingEvolutionaryGuideTests(unittest.TestCase):
    def execute(self, filename: str, fence: str) -> dict[str, object]:
        path = (CASES / filename).with_suffix(".py")
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        return scope

    def test_sobol_sampled_candidate(self) -> None:
        scope = self.execute("06_derivative_free.md", "```")
        scores = scope["acqf"](scope["samples"])
        torch.testing.assert_close(scope["score"], scores.max())

    def test_nsga2_pareto_candidates(self) -> None:
        scope = self.execute("07_evolutionary.md", "~~~")
        points = scope["pareto_y"]
        dominates = (points[:, None, :] >= points[None, :, :]).all(dim=-1)
        dominates &= (points[:, None, :] > points[None, :, :]).any(dim=-1)
        self.assertFalse(dominates.any().item())


if __name__ == "__main__":
    unittest.main()

"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "optimization/05_constraints.py"


class OptimizationConstraintsGuideTests(unittest.TestCase):
    def test_linear_candidate_feasibility_example(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        candidate = scope["candidate"]
        self.assertEqual(tuple(candidate.shape), (1, 1))
        self.assertGreaterEqual(candidate[0, 0].item(), 0.3 - 1e-6)
        self.assertLessEqual(candidate[0, 0].item(), 1.0 + 1e-6)
        self.assertTrue(torch.isfinite(scope["value"]).all())


if __name__ == "__main__":
    unittest.main()

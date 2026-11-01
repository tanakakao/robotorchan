"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "optimization/02_gradient_based.py"


class OptimizationGuideTests(unittest.TestCase):
    def test_gradient_optimizer_dispatch_example(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        candidate = scope["candidate"]
        bounds = scope["bounds"]
        self.assertEqual(tuple(candidate.shape), (1, 1))
        self.assertTrue(torch.all(candidate >= bounds[0]))
        self.assertTrue(torch.all(candidate <= bounds[1]))


if __name__ == "__main__":
    unittest.main()

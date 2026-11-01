"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "optimization/11_hybrid.py"


class HybridOptimizationGuideTests(unittest.TestCase):
    def test_de_to_botorch_local_candidate(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        self.assertEqual(tuple(scope["candidate"].shape), (1, 1))
        self.assertTrue(torch.isfinite(scope["score"]).all())


if __name__ == "__main__":
    unittest.main()

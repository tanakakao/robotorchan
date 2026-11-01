"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

CASES = Path(__file__).with_name("runtime_cases") / "12_hierarchical_contextual_gp.py"


class HierarchicalGuideExampleTests(unittest.TestCase):
    def test_mixed_hierarchical_example(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        model = scope["model"]
        self.assertEqual(model.cat_dims, (0,))
        self.assertTrue(torch.equal(model.raw_train_X, scope["train_X"]))
        self.assertIsInstance(model.make_mll(), ExactMarginalLogLikelihood)


if __name__ == "__main__":
    unittest.main()

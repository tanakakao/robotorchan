"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "optimization/04_mixed_discrete.py"


class MixedDiscreteGuideTests(unittest.TestCase):
    def test_fixed_feature_enumeration_example(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        candidate = scope["candidate"]
        self.assertEqual(tuple(candidate.shape), (1, 2))
        self.assertIn(candidate[0, 1].item(), (0.0, 1.0))
        self.assertTrue(torch.all(candidate >= scope["bounds"][0]))
        self.assertTrue(torch.all(candidate <= scope["bounds"][1]))


if __name__ == "__main__":
    unittest.main()

"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "optimization/09_high_dimensional.py"


class HighDimensionalGuideTests(unittest.TestCase):
    def test_hesbo_embedding_and_original_candidates(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        self.assertEqual(tuple(scope["original"].shape), (1, 6))
        self.assertEqual(tuple(scope["embedded"].shape), (1, 2))
        torch.testing.assert_close(scope["strategy"].project(scope["embedded"]), scope["original"])


if __name__ == "__main__":
    unittest.main()

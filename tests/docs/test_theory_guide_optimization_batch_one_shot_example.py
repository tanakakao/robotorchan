"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "optimization/08_batch_one_shot.py"


class BatchOneShotGuideTests(unittest.TestCase):
    def test_qkg_augmented_and_experimental_candidates(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        self.assertEqual(scope["augmented_q"], 6)
        self.assertEqual(tuple(scope["candidates"].shape), (1, 2, 1))
        self.assertTrue(torch.equal(scope["candidates"], scope["augmented"][:, :2, :]))


if __name__ == "__main__":
    unittest.main()

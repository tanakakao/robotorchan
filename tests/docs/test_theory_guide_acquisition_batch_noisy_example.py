"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "acquisition/04_batch_noisy.py"


class BatchNoisyGuideExampleTests(unittest.TestCase):
    def test_qlognei_batch_shape_and_finiteness(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        self.assertEqual(tuple(scope["candidates"].shape), (2, 2, 1))
        self.assertEqual(tuple(scope["scores"].shape), (2,))
        self.assertTrue(torch.isfinite(scope["scores"]).all())
        self.assertEqual(tuple(scope["acqf"].X_baseline.shape), (8, 1))


if __name__ == "__main__":
    unittest.main()

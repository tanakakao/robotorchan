"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "acquisition/08_constraints.py"


class ConstrainedAcquisitionGuideExampleTests(unittest.TestCase):
    def test_continuous_constrained_qei_example(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        self.assertEqual(tuple(scope["train_y"].shape), (3, 2))
        self.assertEqual(tuple(scope["value"].shape), (1,))
        self.assertTrue(torch.isfinite(scope["value"]).all())


if __name__ == "__main__":
    unittest.main()

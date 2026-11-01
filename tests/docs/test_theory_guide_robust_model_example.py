"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "14_robust_gaussian_process.py"


class RobustGuideExampleTests(unittest.TestCase):
    def test_student_t_guide_example(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        model = scope["model"]
        loss = scope["loss"]
        posterior = scope["posterior"]
        self.assertFalse(model.supports_mll)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(posterior.mean).all())
        self.assertEqual(tuple(posterior.mean.shape[-2:]), (2, 1))
        with self.assertRaisesRegex(RuntimeError, "training_loss"):
            model.make_mll()


if __name__ == "__main__":
    unittest.main()

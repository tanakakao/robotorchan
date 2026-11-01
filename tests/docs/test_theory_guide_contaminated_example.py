"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "14_robust_gaussian_process_contamination.py"


class ContaminatedGuideExampleTests(unittest.TestCase):
    def test_contamination_snippet_and_training_contract(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        model = scope["model"]
        self.assertFalse(model.supports_mll)
        self.assertTrue(torch.isfinite(scope["loss"]))
        self.assertTrue(torch.isfinite(scope["posterior"].mean).all())
        probabilities = scope["diagnostic"]
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(torch.all((probabilities >= 0) & (probabilities <= 1)))
        with self.assertRaisesRegex(RuntimeError, "training_loss"):
            model.make_mll()


if __name__ == "__main__":
    unittest.main()

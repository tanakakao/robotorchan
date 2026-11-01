"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "15_heteroskedastic_noise.py"


class HeteroskedasticGuideExampleTests(unittest.TestCase):
    def test_joint_model_response_and_noise_contract(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        model = scope["model"]
        self.assertFalse(model.supports_mll)
        self.assertTrue(torch.isfinite(scope["loss"]))
        self.assertTrue(torch.isfinite(scope["response"].mean).all())
        self.assertTrue(torch.isfinite(scope["log_noise"].mean).all())
        variance = scope["observation_variance"]
        self.assertTrue(torch.isfinite(variance).all())
        self.assertTrue(torch.all(variance > 0))
        with self.assertRaisesRegex(RuntimeError, "training_loss"):
            model.make_mll()


if __name__ == "__main__":
    unittest.main()

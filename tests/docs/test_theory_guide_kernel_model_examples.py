"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases")


class KernelModelGuideExamplesTests(unittest.TestCase):
    def run_example(self, chapter: str) -> dict[str, object]:
        path = (CASES / chapter).with_suffix(".py")
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        return scope

    def test_uncertain_input_posterior(self) -> None:
        scope = self.run_example("16_uncertain_input_gp.md")
        model = scope["model"]
        posterior = scope["posterior"]
        self.assertEqual(tuple(posterior.mean.shape[-2:]), (2, 1))
        self.assertTrue(torch.isfinite(posterior.mean).all())
        self.assertTrue(torch.isfinite(posterior.variance).all())
        self.assertTrue(torch.equal(model.raw_train_X_std, scope["train_X_std"]))

    def test_nonstationary_local_lengthscale(self) -> None:
        scope = self.run_example("17_nonstationary_gp.md")
        model = scope["model"]
        X = scope["X"]
        lengthscales = model.gibbs_kernel.local_lengthscale(X)
        self.assertEqual(tuple(lengthscales.shape), tuple(X.shape))
        self.assertTrue(torch.isfinite(lengthscales).all())
        self.assertTrue(torch.all(lengthscales > 0))


if __name__ == "__main__":
    unittest.main()

"""Run uncertain-input and nonstationary examples published in the model guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide"


class KernelModelGuideExamplesTests(unittest.TestCase):
    def run_example(self, chapter: str) -> dict[str, object]:
        path = GUIDE / chapter
        match = re.search(
            r"```python\n(.*?)\n```",
            path.read_text(encoding="utf-8"),
            re.DOTALL,
        )
        self.assertIsNotNone(match, f"Missing Python example in {chapter}")
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(path), "exec"), scope)
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

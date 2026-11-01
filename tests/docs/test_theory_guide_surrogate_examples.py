"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases")


class SurrogateGuideExamplesTests(unittest.TestCase):
    def run_example(self, chapter: str) -> dict[str, object]:
        path = (CASES / chapter).with_suffix(".py")
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        return scope

    def test_pca_gp(self) -> None:
        scope = self.run_example("18_dimensionality_reduction_gp.md")
        posterior = scope["posterior"]
        self.assertEqual(tuple(posterior.mean.shape[-2:]), (2, 1))
        self.assertTrue(torch.isfinite(posterior.mean).all())

    def test_random_forest(self) -> None:
        scope = self.run_example("23_non_gp_surrogates.md")
        posterior = scope["posterior"]
        self.assertEqual(tuple(posterior.mean.shape[-2:]), (2, 1))
        self.assertTrue(torch.isfinite(posterior.mean).all())

    def test_binary_classifier(self) -> None:
        scope = self.run_example("24_classification.md")
        probabilities = scope["class_probabilities"]
        self.assertEqual(probabilities.shape[-1], 2)
        self.assertTrue(torch.isfinite(probabilities).all())
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones_like(probabilities[..., 0]),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertTrue(torch.isfinite(scope["latent_posterior"].mean).all())


if __name__ == "__main__":
    unittest.main()

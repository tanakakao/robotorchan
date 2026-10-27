"""Exercise PCA, random-forest, and binary-classifier guide snippets."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/development/theory-implementation/guide"


class SurrogateGuideExamplesTests(unittest.TestCase):
    def run_example(self, chapter: str) -> dict[str, object]:
        path = GUIDE / chapter
        match = re.search(
            r"```python\n(.*?)\n```",
            path.read_text(encoding="utf-8"),
            re.DOTALL,
        )
        self.assertIsNotNone(match, f"Missing Python example: {chapter}")
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(path), "exec"), scope)
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

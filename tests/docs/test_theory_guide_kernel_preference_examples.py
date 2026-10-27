"""Execute kernel and preference-learning Python examples from their guide chapters."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/development/theory-implementation/guide"


class KernelPreferenceGuideExamplesTests(unittest.TestCase):
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

    def test_kernel_guide_fitted_gp(self) -> None:
        scope = self.run_example("03_kernel.md")
        model = scope["model"]
        X = scope["X"]
        posterior = model.posterior(X[:2])
        self.assertEqual(tuple(posterior.mean.shape[-2:]), (2, 1))
        self.assertTrue(torch.isfinite(posterior.mean).all())

    def test_preference_learning_guide_mll(self) -> None:
        scope = self.run_example("10_preference_learning.md")
        model = scope["model"]
        self.assertEqual(model.datapoints.shape, (3, 1))
        self.assertEqual(model.comparisons.shape, (2, 2))
        self.assertIsNotNone(model.make_mll())


if __name__ == "__main__":
    unittest.main()

"""Execute published fidelity, multitask, and structured-output guide snippets."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/development/theory-implementation/guide"


class StructuredModelGuideExamplesTests(unittest.TestCase):
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

    def test_multi_fidelity_posterior(self) -> None:
        scope = self.run_example("06_multi_fidelity.md")
        model = scope["model"]
        X = scope["X"]
        posterior = model.posterior(X[:2])
        self.assertEqual(tuple(posterior.mean.shape[-2:]), (2, 1))
        self.assertTrue(torch.isfinite(posterior.mean).all())

    def test_multitask_posterior(self) -> None:
        scope = self.run_example("07_multitask_multioutput.md")
        model = scope["model"]
        X = scope["X"]
        posterior = model.posterior(X[:2])
        self.assertTrue(torch.isfinite(posterior.mean).all())
        self.assertEqual(posterior.mean.shape[-2], 2)

    def test_higher_order_constructor(self) -> None:
        scope = self.run_example("11_structured_output.md")
        self.assertEqual(scope["X"].shape, (3, 1))
        self.assertEqual(scope["Y"].shape, (3, 2, 2))
        self.assertEqual(type(scope["model"]).__name__, "HigherOrderGP")


if __name__ == "__main__":
    unittest.main()

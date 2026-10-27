"""Exercise published core-model examples without duplicating their Python source."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/development/theory-implementation/guide"


class CoreModelGuideExamplesTests(unittest.TestCase):
    def _run_first_python_block(self, chapter: str) -> dict[str, object]:
        path = GUIDE / chapter
        text = path.read_text(encoding="utf-8")
        match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
        self.assertIsNotNone(match, f"Missing Python example in {chapter}")
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(path), "exec"), scope)
        return scope

    def test_exact_gp_posterior(self) -> None:
        scope = self._run_first_python_block("02_gaussian_process.md")
        posterior = scope["p"]
        self.assertEqual(tuple(posterior.mean.shape[-2:]), (2, 1))
        self.assertTrue(torch.isfinite(posterior.mean).all())
        self.assertTrue(torch.isfinite(posterior.variance).all())

    def test_mixed_gp_posterior(self) -> None:
        scope = self._run_first_python_block("05_mixed_variables.md")
        model = scope["model"]
        X = scope["X"]
        posterior = model.posterior(X[:2])
        self.assertEqual(tuple(posterior.mean.shape[-2:]), (2, 1))
        self.assertTrue(torch.isfinite(posterior.mean).all())

    def test_variational_gp_elbo(self) -> None:
        scope = self._run_first_python_block("09_variational_gp.md")
        model = scope["model"]
        elbo = scope["elbo"]
        self.assertEqual(type(elbo).__name__, "VariationalELBO")
        self.assertEqual(model.raw_train_X.shape[-2], 30)


if __name__ == "__main__":
    unittest.main()

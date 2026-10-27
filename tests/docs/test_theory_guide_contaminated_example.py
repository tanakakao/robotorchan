"""Execute the contamination-mixture example in the theory implementation guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/theory-implementation/guide/14_robust_gaussian_process.md"
)


class ContaminatedGuideExampleTests(unittest.TestCase):
    def test_contamination_snippet_and_training_contract(self) -> None:
        section = GUIDE.read_text(encoding="utf-8").split("## Contamination Mixture", 1)[1]
        match = re.search(r"~~~python\n(.*?)\n~~~", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
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

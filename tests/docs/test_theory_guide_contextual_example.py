"""Execute the contextual multi-output snippet from the theory guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide/12_hierarchical_contextual_gp.md"


class ContextualGuideExampleTests(unittest.TestCase):
    def test_mixed_lcemgp_snippet(self) -> None:
        section = GUIDE.read_text(encoding="utf-8").split("### Contextual Multi-Output", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match, "MixedLCEMGP example not found")
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        model = scope["model"]
        self.assertEqual(model.cat_dims, (1,))
        self.assertEqual(model.raw_task_feature, 2)
        self.assertTrue(torch.equal(model.raw_train_X, scope["train_X"]))
        self.assertIsInstance(model.make_mll(), ExactMarginalLogLikelihood)


if __name__ == "__main__":
    unittest.main()

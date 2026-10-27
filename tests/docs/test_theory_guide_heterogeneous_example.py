"""Run the heterogeneous-model example published in the theory implementation guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch


GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/theory-implementation/guide/07_multitask_multioutput.md"
)


class HeterogeneousGuideExampleTests(unittest.TestCase):
    def test_example_runs_without_undefined_variables(self) -> None:
        content = GUIDE.read_text(encoding="utf-8")
        section = content.split("### 異種モデルの出力所有権を確認する", 1)[1]
        match = re.search(r"~~~python\n(.*?)\n~~~", section, re.DOTALL)
        self.assertIsNotNone(match, "HeterogeneousModel example not found")
        assert match is not None

        # Execute the published example itself, not a separately maintained copy.
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)

        model = scope["models"]
        self.assertEqual(model.num_outputs, 2)
        self.assertEqual(model.output_owner(1), (1, 0))
        probabilities = scope["pass_probs"]
        self.assertEqual(tuple(probabilities.shape), (2,))
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=-1),
                torch.ones_like(probabilities[..., 0]),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()

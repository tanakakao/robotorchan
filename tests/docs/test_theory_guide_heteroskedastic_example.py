"""Execute joint heteroskedastic GP code as published in the Japanese guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/theory-implementation/guide/15_heteroskedastic_noise.md"
)


class HeteroskedasticGuideExampleTests(unittest.TestCase):
    def test_joint_model_response_and_noise_contract(self) -> None:
        match = re.search(
            r"```python\n(.*?)\n```",
            GUIDE.read_text(encoding="utf-8"),
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        model = scope["model"]
        self.assertFalse(model.supports_mll)
        self.assertTrue(torch.isfinite(scope["loss"]))
        self.assertTrue(torch.isfinite(scope["response"].mean).all())
        self.assertTrue(torch.isfinite(scope["log_noise"].mean).all())
        variance = scope["observation_variance"]
        self.assertTrue(torch.isfinite(variance).all())
        self.assertTrue(torch.all(variance > 0))
        with self.assertRaisesRegex(RuntimeError, "training_loss"):
            model.make_mll()


if __name__ == "__main__":
    unittest.main()

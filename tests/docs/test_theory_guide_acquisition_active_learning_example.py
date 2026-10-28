"""Run regression active-learning code from the Japanese acquisition guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide/acquisition/09_active_learning.md"


class ActiveLearningGuideTests(unittest.TestCase):
    def test_variance_and_std_match_posterior(self) -> None:
        section = GUIDE.read_text(encoding="utf-8").split("## 6.", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        model = scope["model"]
        candidates = scope["candidates"]
        variance = model.posterior(candidates).variance[..., 0, 0]
        torch.testing.assert_close(scope["variance_score"], variance)
        torch.testing.assert_close(scope["std_score"], variance.clamp_min(0).sqrt())


if __name__ == "__main__":
    unittest.main()

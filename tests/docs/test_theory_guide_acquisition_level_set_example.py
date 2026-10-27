"""Execute the level-set Straddle example and check the moment formula."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/theory-implementation/guide/acquisition/10_level_set.md"
)


class LevelSetGuideExampleTests(unittest.TestCase):
    def test_straddle_matches_posterior_formula(self) -> None:
        section = GUIDE.read_text(encoding="utf-8").split("## 6.", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        posterior = scope["model"].posterior(scope["candidates"])
        mean = posterior.mean[..., 0, 0]
        std = posterior.variance[..., 0, 0].clamp_min(0).sqrt()
        reference = scope["beta"] * std - (mean - scope["target"]).abs()
        torch.testing.assert_close(scope["scores"], reference, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

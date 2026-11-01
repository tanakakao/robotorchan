"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "acquisition/09_active_learning.py"


class ActiveLearningGuideTests(unittest.TestCase):
    def test_variance_and_std_match_posterior(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        model = scope["model"]
        candidates = scope["candidates"]
        variance = model.posterior(candidates).variance[..., 0, 0]
        torch.testing.assert_close(scope["variance_score"], variance)
        torch.testing.assert_close(scope["std_score"], variance.clamp_min(0).sqrt())


if __name__ == "__main__":
    unittest.main()

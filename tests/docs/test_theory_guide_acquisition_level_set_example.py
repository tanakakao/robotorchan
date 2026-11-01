"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "acquisition/10_level_set.py"


class LevelSetGuideExampleTests(unittest.TestCase):
    def test_straddle_matches_posterior_formula(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        posterior = scope["model"].posterior(scope["candidates"])
        mean = posterior.mean[..., 0, 0]
        std = posterior.variance[..., 0, 0].clamp_min(0).sqrt()
        reference = scope["beta"] * std - (mean - scope["target"]).abs()
        torch.testing.assert_close(scope["scores"], reference, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

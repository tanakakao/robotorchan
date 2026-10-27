"""Execute acquisition guide snippets and compare analytic formulas independently."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch
from torch.distributions import Normal

GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/theory-implementation/guide/acquisition"
)


class AcquisitionGuideExamplesTests(unittest.TestCase):
    def run_example(self, chapter: str) -> dict[str, object]:
        path = GUIDE / chapter
        section = path.read_text(encoding="utf-8").split("## 6.", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(path), "exec"), scope)
        return scope

    def test_log_expected_improvement_reference(self) -> None:
        scope = self.run_example("02_improvement.md")
        posterior = scope["model"].posterior(scope["candidates"].squeeze(-2))
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.squeeze(-1).sqrt()
        normal = Normal(torch.zeros_like(mean), torch.ones_like(mean))
        improvement = mean - scope["best_f"]
        z = improvement / std
        reference = improvement * normal.cdf(z) + std * torch.exp(normal.log_prob(z))
        torch.testing.assert_close(scope["scores"].exp(), reference, rtol=1e-5, atol=1e-7)

    def test_upper_confidence_bound_reference(self) -> None:
        scope = self.run_example("03_confidence_bound.md")
        posterior = scope["model"].posterior(scope["candidates"].squeeze(-2))
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.squeeze(-1).sqrt()
        reference = mean + scope["beta"] ** 0.5 * std
        torch.testing.assert_close(scope["scores"], reference, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

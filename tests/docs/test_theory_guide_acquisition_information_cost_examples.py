"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "acquisition"


class InformationCostGuideTests(unittest.TestCase):
    def execute(self, chapter: str, fence: str) -> dict[str, object]:
        path = (CASES / chapter).with_suffix(".py")
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        return scope

    def test_mes_gibbon_guide_example(self) -> None:
        scope = self.execute("05_information_theoretic.md", "~~~")
        self.assertTrue(torch.isfinite(scope["mes_value"]).all())
        self.assertTrue(torch.isfinite(scope["gibbon_value"]).all())
        self.assertEqual(tuple(scope["candidate_set"].shape), (32, 1))

    def test_fidelity_cost_guide_example(self) -> None:
        scope = self.execute("11_multifidelity_cost_aware.md", "```")
        costs = scope["costs"].reshape(-1)
        self.assertEqual(costs.numel(), 2)
        torch.testing.assert_close(costs, torch.tensor([0.6, 1.1], dtype=torch.double))


if __name__ == "__main__":
    unittest.main()

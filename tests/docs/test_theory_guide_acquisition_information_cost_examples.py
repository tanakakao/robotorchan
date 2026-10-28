"""Execute the information-theoretic and fidelity-cost guide snippets."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide/acquisition"


class InformationCostGuideTests(unittest.TestCase):
    def execute(self, chapter: str, fence: str) -> dict[str, object]:
        path = GUIDE / chapter
        match = re.search(
            rf"{re.escape(fence)}python\n(.*?)\n{re.escape(fence)}",
            path.read_text(encoding="utf-8"),
            re.DOTALL,
        )
        self.assertIsNotNone(match, f"No {fence}python example found in {chapter}")
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(path), "exec"), scope)
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

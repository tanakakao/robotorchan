"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases")


class HighDimensionalGuideExamplesTests(unittest.TestCase):
    def run_example(self, chapter: str) -> dict[str, object]:
        path = (CASES / chapter).with_suffix(".py")
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        return scope

    def test_turbo_strategy_construction(self) -> None:
        scope = self.run_example("20_high_dimensional_search.md")
        strategy = scope["strategy"]
        self.assertTrue(strategy.state.length > 0)
        self.assertEqual(tuple(scope["bounds"].shape), (2, 2))

    def test_additive_map_saas_construction(self) -> None:
        scope = self.run_example("21_advanced_high_dimensional_models.md")
        model = scope["model"]
        self.assertEqual(tuple(model.raw_train_X.shape), (12, 6))
        self.assertEqual(tuple(model.raw_train_Y.shape), (12, 1))


if __name__ == "__main__":
    unittest.main()

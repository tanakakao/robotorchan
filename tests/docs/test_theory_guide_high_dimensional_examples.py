"""Execute high-dimensional search and MAP-SAAS guide snippets."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/development/theory-implementation/guide"


class HighDimensionalGuideExamplesTests(unittest.TestCase):
    def run_example(self, chapter: str) -> dict[str, object]:
        path = GUIDE / chapter
        match = re.search(
            r"```python\n(.*?)\n```",
            path.read_text(encoding="utf-8"),
            re.DOTALL,
        )
        self.assertIsNotNone(match, f"Missing Python example: {chapter}")
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(path), "exec"), scope)
        return scope

    def test_turbo_strategy_construction(self) -> None:
        scope = self.run_example("20_high_dimensional_search.md")
        strategy = scope["strategy"]
        self.assertTrue(0 < strategy.state.length)
        self.assertEqual(tuple(scope["bounds"].shape), (2, 2))

    def test_additive_map_saas_construction(self) -> None:
        scope = self.run_example("21_advanced_high_dimensional_models.md")
        model = scope["model"]
        self.assertEqual(tuple(model.raw_train_X.shape), (12, 6))
        self.assertEqual(tuple(model.raw_train_Y.shape), (12, 1))


if __name__ == "__main__":
    unittest.main()

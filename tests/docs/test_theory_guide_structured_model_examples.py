"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases")


class StructuredModelGuideExamplesTests(unittest.TestCase):
    def run_example(self, chapter: str) -> dict[str, object]:
        path = (CASES / chapter).with_suffix(".py")
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        return scope

    def test_higher_order_constructor(self) -> None:
        scope = self.run_example("11_structured_output.md")
        self.assertEqual(scope["X"].shape, (3, 1))
        self.assertEqual(scope["Y"].shape, (3, 2, 2))
        self.assertEqual(type(scope["model"]).__name__, "HigherOrderGP")


if __name__ == "__main__":
    unittest.main()

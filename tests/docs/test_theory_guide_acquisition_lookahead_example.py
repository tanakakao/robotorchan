"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases") / "acquisition/06_lookahead.py"


class LookaheadGuideTests(unittest.TestCase):
    def test_qkg_augmented_candidates_example(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        self.assertEqual(scope["augmented_q"], 17)
        self.assertEqual(tuple(scope["experimental_candidates"].shape), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()

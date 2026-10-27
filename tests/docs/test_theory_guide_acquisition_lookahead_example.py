"""Execute the one-shot Knowledge Gradient example from the Japanese guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/theory-implementation/guide/acquisition/06_lookahead.md"
)


class LookaheadGuideTests(unittest.TestCase):
    def test_qkg_augmented_candidates_example(self) -> None:
        chapter = GUIDE.read_text(encoding="utf-8")
        match = re.search(r"~~~python\n(.*?)\n~~~", chapter, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        self.assertEqual(scope["augmented_q"], 17)
        self.assertEqual(tuple(scope["experimental_candidates"].shape), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()

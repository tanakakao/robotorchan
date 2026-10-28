"""Execute the constrained qEI example published in the Japanese acquisition guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide/acquisition/08_constraints.md"


class ConstrainedAcquisitionGuideExampleTests(unittest.TestCase):
    def test_continuous_constrained_qei_example(self) -> None:
        chapter = GUIDE.read_text(encoding="utf-8")
        section = chapter.split("### 4.2", 1)[1].split("### 4.3", 1)[0]
        match = re.search(r"~~~python\n(.*?)\n~~~", section, re.DOTALL)
        self.assertIsNotNone(match, "Missing continuous constrained qEI example")
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        self.assertEqual(tuple(scope["train_y"].shape), (3, 2))
        self.assertEqual(tuple(scope["value"].shape), (1,))
        self.assertTrue(torch.isfinite(scope["value"]).all())


if __name__ == "__main__":
    unittest.main()

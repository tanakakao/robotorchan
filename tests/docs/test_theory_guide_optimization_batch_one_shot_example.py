"""Execute the qKG batch/one-shot candidate shape example in the guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide/optimization/08_batch_one_shot.md"


class BatchOneShotGuideTests(unittest.TestCase):
    def test_qkg_augmented_and_experimental_candidates(self) -> None:
        chapter = GUIDE.read_text(encoding="utf-8")
        section = chapter.split("## 6.", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        self.assertEqual(scope["augmented_q"], 6)
        self.assertEqual(tuple(scope["candidates"].shape), (1, 2, 1))
        self.assertTrue(torch.equal(scope["candidates"], scope["augmented"][:, :2, :]))


if __name__ == "__main__":
    unittest.main()

"""Run the published qLogNEI batch/noisy acquisition example."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/theory-implementation/guide/acquisition/04_batch_noisy.md"
)


class BatchNoisyGuideExampleTests(unittest.TestCase):
    def test_qlognei_batch_shape_and_finiteness(self) -> None:
        section = GUIDE.read_text(encoding="utf-8").split("## 6.", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        self.assertEqual(tuple(scope["candidates"].shape), (2, 2, 1))
        self.assertEqual(tuple(scope["scores"].shape), (2,))
        self.assertTrue(torch.isfinite(scope["scores"]).all())
        self.assertEqual(tuple(scope["acqf"].X_baseline.shape), (8, 1))


if __name__ == "__main__":
    unittest.main()

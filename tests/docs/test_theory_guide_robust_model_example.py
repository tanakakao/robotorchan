"""Execute the Student-t robust GP example directly from the guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide/14_robust_gaussian_process.md"


class RobustGuideExampleTests(unittest.TestCase):
    def test_student_t_guide_example(self) -> None:
        match = re.search(
            r"```python\n(.*?)\n```",
            GUIDE.read_text(encoding="utf-8"),
            re.DOTALL,
        )
        self.assertIsNotNone(match, "Missing Student-t code example")
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        model = scope["model"]
        loss = scope["loss"]
        posterior = scope["posterior"]
        self.assertFalse(model.supports_mll)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(posterior.mean).all())
        self.assertEqual(tuple(posterior.mean.shape[-2:]), (2, 1))
        with self.assertRaisesRegex(RuntimeError, "training_loss"):
            model.make_mll()


if __name__ == "__main__":
    unittest.main()

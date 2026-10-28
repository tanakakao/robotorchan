"""Execute neural-reduction and expressive-GP guide examples."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide"


class JointEncoderGuideExamplesTests(unittest.TestCase):
    def test_joint_encoder_examples(self) -> None:
        for chapter in ("19_neural_reduction_gp.md", "22_expressive_gp.md"):
            with self.subTest(chapter=chapter):
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
                model = scope["model"]
                self.assertEqual(model.latent_dim, 2)


if __name__ == "__main__":
    unittest.main()

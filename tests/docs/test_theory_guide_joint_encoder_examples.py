"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

CASES = Path(__file__).with_name("runtime_cases")


class JointEncoderGuideExamplesTests(unittest.TestCase):
    def test_joint_encoder_examples(self) -> None:
        for chapter in ("19_neural_reduction_gp.md", "22_expressive_gp.md"):
            with self.subTest(chapter=chapter):
                path = (CASES / chapter).with_suffix(".py")
                scope: dict[str, object] = {"__name__": "__runtime_case__"}
                with torch.random.fork_rng():
                    torch.manual_seed(0)
                    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
                model = scope["model"]
                self.assertEqual(model.latent_dim, 2)


if __name__ == "__main__":
    unittest.main()

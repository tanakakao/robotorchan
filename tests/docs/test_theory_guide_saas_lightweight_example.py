"""Execute the lightweight SAAS GP example without NUTS sampling."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest
import torch

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide/08_high_dimensional_gp.md"


class SaasGuideExampleTests(unittest.TestCase):
    def test_lightweight_constructor_example(self) -> None:
        pytest.importorskip("numpyro")
        chapter = GUIDE.read_text(encoding="utf-8")
        section = chapter.split("MCMCを実行しない基本契約の確認", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        model = scope["model"]
        self.assertFalse(model.supports_mll)
        self.assertTrue(torch.equal(model.raw_train_X, scope["X"]))
        self.assertTrue(torch.equal(model.raw_train_Y, scope["Y"]))


if __name__ == "__main__":
    unittest.main()

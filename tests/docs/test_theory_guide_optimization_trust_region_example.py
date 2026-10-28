"""Execute TuRBO state transitions from the Japanese optimization guide."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[2] / "docs/guide/optimization/10_trust_region.md"


class TrustRegionGuideTests(unittest.TestCase):
    def test_turbo_success_and_failure_length_updates(self) -> None:
        chapter = GUIDE.read_text(encoding="utf-8")
        section = chapter.split("## 5.", 1)[1]
        match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        scope: dict[str, object] = {"__name__": "__guide_example__"}
        exec(compile(match.group(1), str(GUIDE), "exec"), scope)
        state = scope["state"]
        self.assertEqual(state.length, 0.4)
        self.assertEqual(state.best_value, 0.7)
        self.assertEqual(state.success_counter, 0)
        self.assertEqual(state.failure_counter, 0)


if __name__ == "__main__":
    unittest.main()

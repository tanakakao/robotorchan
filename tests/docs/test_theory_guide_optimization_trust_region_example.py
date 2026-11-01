"""Preserve runtime and numerical checks independently of Guide prose."""

from __future__ import annotations

import unittest
from pathlib import Path

CASES = Path(__file__).with_name("runtime_cases") / "optimization/10_trust_region.py"


class TrustRegionGuideTests(unittest.TestCase):
    def test_turbo_success_and_failure_length_updates(self) -> None:
        scope: dict[str, object] = {"__name__": "__runtime_case__"}
        exec(compile(CASES.read_text(encoding="utf-8"), str(CASES), "exec"), scope)
        state = scope["state"]
        self.assertEqual(state.length, 0.4)
        self.assertEqual(state.best_value, 0.7)
        self.assertEqual(state.success_counter, 0)
        self.assertEqual(state.failure_counter, 0)


if __name__ == "__main__":
    unittest.main()

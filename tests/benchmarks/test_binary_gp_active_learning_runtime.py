"""Minimal real GP refit and acquisition integration test for Phase 5 AL."""

from __future__ import annotations

import importlib
import math
from pathlib import Path


def test_binary_gp_active_learning_real_closed_loop(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
    monkeypatch.syspath_prepend(str(examples))
    smoke = importlib.import_module("binary_gp_active_learning_smoke")
    multiseed = importlib.import_module("binary_gp_active_learning_multiseed")

    # No surrogate/acquisition stubs: fit all three real GP classifiers,
    # select one candidate, and refit all three models.
    rows = smoke.run(seed=0, budget=1, epochs=1)
    assert len(rows) == 6
    assert {(row["strategy"], row["labels_added"]) for row in rows} == {
        (strategy, step)
        for strategy in ("random", "sobol", "predictive_entropy")
        for step in (0, 1)
    }
    for row in rows:
        assert math.isfinite(row["accuracy"])
        assert math.isfinite(row["brier"])
        assert 0 <= row["accuracy"] <= 1
        assert 0 <= row["brier"] <= 1

    # Check that the real integration path also produces a complete archive.
    output = tmp_path / "real-gp-al"
    manifest = multiseed.execute(
        output, git_sha="test-runtime", seeds=1, budget=1, epochs=1
    )
    assert manifest["status"] == "complete"
    assert manifest["completed_seeds"] == [0]
    assert manifest["failed_seeds"] == []
    assert (output / "curves.csv").is_file()

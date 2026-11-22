"""Minimal real GP refit and acquisition integration test for Phase 5 AL."""

from __future__ import annotations

import csv
import importlib
import math
from pathlib import Path


def test_binary_gp_active_learning_real_closed_loop(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
    monkeypatch.syspath_prepend(str(examples))
    multiseed = importlib.import_module("binary_gp_active_learning_multiseed")

    # No surrogate/acquisition stubs: fit all three real GP classifiers,
    # select one candidate, and refit all three models.
    output = tmp_path / "real-gp-al"
    manifest = multiseed.execute(output, git_sha="test-runtime", seeds=1, budget=1, epochs=1)
    assert manifest["status"] == "complete"
    assert manifest["completed_seeds"] == [0]
    assert manifest["failed_seeds"] == []
    with (output / "curves.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {(row["strategy"], int(row["labels_added"])) for row in rows} == {
        (strategy, step)
        for strategy in ("random", "sobol", "predictive_entropy")
        for step in (0, 1)
    }
    for row in rows:
        accuracy = float(row["accuracy"])
        brier = float(row["brier"])
        assert math.isfinite(accuracy) and 0 <= accuracy <= 1
        assert math.isfinite(brier) and 0 <= brier <= 1

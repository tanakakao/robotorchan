"""Focused archive-contract test; no expensive GP refits in CI."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path


def test_gp_al_multiseed_archive_excludes_incomplete_pairs(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
    monkeypatch.syspath_prepend(str(examples))
    runner = importlib.import_module("binary_gp_active_learning_multiseed")

    def fake_run(*, seed, budget, epochs):
        assert budget == 1
        assert epochs == 2
        if seed == 1:
            raise RuntimeError("synthetic failure")
        return [
            {
                "strategy": strategy,
                "seed": seed,
                "labels_added": step,
                "accuracy": accuracy,
                "brier": brier,
            }
            for step in range(budget + 1)
            for strategy, accuracy, brier in (
                ("random", 0.5, 0.25),
                ("sobol", 0.6, 0.20),
                ("predictive_entropy", 0.7, 0.15),
            )
        ]

    monkeypatch.setattr(runner, "run", fake_run)
    root = tmp_path / "evidence"
    manifest = runner.execute(root, git_sha="test-commit", seeds=2, budget=1, epochs=2)
    assert manifest["status"] == "partial_failure"
    assert manifest["completed_seeds"] == [0]
    assert manifest["failed_seeds"] == [1]
    assert json.loads((root / "manifest.json").read_text()) == manifest

    def read_csv(name):
        with (root / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    curves = read_csv("curves.csv")
    assert len(curves) == 6
    assert {row["seed"] for row in curves} == {"0"}
    differences = read_csv("paired_differences.csv")
    assert len(differences) == 8
    entropy_accuracy = next(
        row
        for row in differences
        if row["strategy"] == "predictive_entropy"
        and row["metric"] == "accuracy"
        and row["labels_added"] == "1"
    )
    assert abs(float(entropy_accuracy["difference_vs_random"]) - 0.2) < 1e-12
    summary = read_csv("summary.csv")
    assert len(summary) == 12
    assert all(row["n"] == "1" and row["std"] == "" for row in summary)
    failures = json.loads((root / "failures.json").read_text())
    assert len(failures) == 1
    assert failures[0]["seed"] == 1
    assert failures[0]["error_type"] == "RuntimeError"
    assert "synthetic failure" in failures[0]["traceback"]

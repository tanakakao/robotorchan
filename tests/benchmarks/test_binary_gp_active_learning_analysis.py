"""Deterministic paired analysis checks, without running expensive GP fits."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import pytest


def test_gp_al_paired_analysis(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
    monkeypatch.syspath_prepend(str(examples))
    analyzer = importlib.import_module("binary_gp_active_learning_analyze")
    manifest = {
        "status": "complete",
        "seeds": [0, 1],
        "completed_seeds": [0, 1],
        "failed_seeds": [],
        "budget": 2,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for seed in (0, 1):
        for strategy, offset in (("random", 0.0), ("sobol", 0.1), ("predictive_entropy", 0.2)):
            for step in range(3):
                rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "labels_added": step,
                        "accuracy": 0.4 + offset * step,
                        "brier": 0.5 - offset * step,
                    }
                )
    with (tmp_path / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results = analyzer.analyze(tmp_path)
    assert len(results) == 8
    entropy_accuracy = next(
        row
        for row in results
        if row["strategy"] == "predictive_entropy"
        and row["metric"] == "accuracy"
        and row["statistic"] == "endpoint"
    )
    assert entropy_accuracy["n"] == 2
    assert entropy_accuracy["mean_difference"] == pytest.approx(0.4)
    entropy_auc = next(
        row
        for row in results
        if row["strategy"] == "predictive_entropy"
        and row["metric"] == "accuracy"
        and row["statistic"] == "normalized_auc"
    )
    assert entropy_auc["mean_difference"] == pytest.approx(0.2)
    manifest["status"] = "partial_failure"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        analyzer.analyze(tmp_path)

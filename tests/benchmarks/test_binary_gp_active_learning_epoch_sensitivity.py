"""Focused checks for paired GP AL fit-epoch comparisons."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import pytest


def test_binary_gp_al_epoch_comparison(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
    monkeypatch.syspath_prepend(str(examples))
    analyzer = importlib.import_module("binary_gp_active_learning_epoch_sensitivity")

    def archive(name, epochs):
        root = tmp_path / name
        root.mkdir()
        manifest = {
            "status": "complete",
            "failed_seeds": [],
            "completed_seeds": [0, 1],
            "seeds": [0, 1],
            "strategies": list(analyzer.STRATEGIES),
            "budget": 1,
            "epochs_per_refit": epochs,
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        rows = [
            {
                "seed": seed,
                "strategy": strategy,
                "labels_added": step,
                "accuracy": 0.6 + (epochs - 8) / 240,
                "brier": 0.2 - (epochs - 8) / 480,
            }
            for seed in (0, 1)
            for strategy in analyzer.STRATEGIES
            for step in (0, 1)
        ]
        with (root / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return root

    low, high = archive("low", 8), archive("high", 32)
    result = analyzer.compare(low, high)
    assert len(result) == 6
    assert all(row["n"] == 2 for row in result)
    accuracy = next(
        row for row in result if row["strategy"] == "predictive_entropy"
        and row["metric"] == "accuracy"
    )
    assert accuracy["mean_change"] == pytest.approx(0.1)
    assert accuracy["sd_change"] == pytest.approx(0)
    manifest_path = high / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["budget"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocols differ"):
        analyzer.compare(low, high)

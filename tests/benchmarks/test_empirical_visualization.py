"""Verify empirical charts cannot be fabricated from missing evidence."""

import json

import pytest

from robotorchan.benchmarks.empirical_visualization import render_empirical_charts


def test_visualization_rejects_missing_summary(tmp_path):
    with pytest.raises(FileNotFoundError):
        render_empirical_charts(tmp_path, tmp_path / "charts")


def test_visualization_rejects_incomplete_summary(tmp_path):
    (tmp_path / "summary.json").write_text(
        json.dumps({"status": "complete", "seed_count": 2}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Incomplete"):
        render_empirical_charts(tmp_path, tmp_path / "charts")
    assert not (tmp_path / "charts").exists()


def test_visualization_rejects_existing_destination(tmp_path):
    destination = tmp_path / "charts"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        render_empirical_charts(tmp_path, destination)


def test_visualization_rejects_corrupted_final_metric(tmp_path):
    """Reject altered CSV metrics before importing the plotting backend."""
    import csv

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    summary = {
        "status": "complete",
        "seed_count": 10,
        "problem": "branin",
        "q": 1,
        "metric": "simple_regret",
        "final_metric_by_seed": {"random": [1.0] * 10},
        "elapsed_seconds_by_seed": {"random": [3.0] * 10},
    }
    (artifacts / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (artifacts / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "problem",
                "q",
                "strategy",
                "seed",
                "total_points",
                "evaluations",
                "metric",
                "value",
            ),
        )
        writer.writeheader()
        for seed in range(10):
            for total in range(1, 71):
                writer.writerow(
                    {
                        "problem": "branin",
                        "q": 1,
                        "strategy": "random",
                        "seed": seed,
                        "total_points": total,
                        "evaluations": max(0, total - 20),
                        "metric": "simple_regret",
                        "value": 2.0 if seed == 0 and total == 70 else 1.0,
                    }
                )
    with (artifacts / "timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("problem", "q", "strategy", "seed"),
        )
        writer.writeheader()
        for seed in range(10):
            writer.writerow({"problem": "branin", "q": 1, "strategy": "random", "seed": seed})
    with pytest.raises(ValueError, match="Curve final value disagrees"):
        render_empirical_charts(artifacts, tmp_path / "charts")
    assert not (tmp_path / "charts").exists()

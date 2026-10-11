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

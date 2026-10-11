"""Test empirical CSV export completeness and overwrite safety."""

import json

import pytest

from robotorchan.benchmarks.empirical_curves import export_empirical_curves


def test_curve_export_rejects_missing_evidence(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_empirical_curves(tmp_path, tmp_path / "curves")


def test_curve_export_rejects_failed_cohort(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime-validated"):
        export_empirical_curves(tmp_path, tmp_path / "curves")
    assert not (tmp_path / "curves").exists()


def test_curve_export_rejects_existing_destination(tmp_path):
    destination = tmp_path / "curves"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        export_empirical_curves(tmp_path, destination)


def test_curve_export_rejects_summary_mismatch(tmp_path, monkeypatch):
    """Do not export inconsistent summary and trajectory evidence."""
    from types import SimpleNamespace

    import torch

    from robotorchan.benchmarks import empirical_curves

    summary = {
        "problem": "branin",
        "q": 1,
        "metric": "simple_regret",
        "final_metric_by_seed": {"random": [2.0] * 10},
        "elapsed_seconds_by_seed": {"random": [3.0] * 10},
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"strategies": ["random"]}), encoding="utf-8"
    )
    run = SimpleNamespace(
        initial_points=20,
        candidate_seconds=(1.0,),
        evaluation_seconds=(1.0,),
        initial_evaluation_seconds=1.0,
    )
    monkeypatch.setattr(empirical_curves, "summarize_empirical_directory", lambda _: summary)
    monkeypatch.setattr(empirical_curves, "load_trajectory", lambda _: run)
    monkeypatch.setattr(empirical_curves, "simple_regret_curve", lambda *_: torch.ones(70))
    monkeypatch.setitem(
        empirical_curves._PROBLEMS,
        "branin",
        lambda: SimpleNamespace(n_objectives=1),
    )
    destination = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="Final curve metric disagrees"):
        export_empirical_curves(tmp_path, destination)
    assert not destination.exists()

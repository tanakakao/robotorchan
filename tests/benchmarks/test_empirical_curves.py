"""Test empirical CSV export completeness and overwrite safety."""

import json

import pytest

from robotorchan.benchmarks.empirical_curves import export_empirical_curves


def test_curve_export_rejects_missing_evidence(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_empirical_curves(tmp_path, tmp_path / "curves")


def test_curve_export_rejects_failed_cohort(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="runtime-validated"):
        export_empirical_curves(tmp_path, tmp_path / "curves")
    assert not (tmp_path / "curves").exists()


def test_curve_export_rejects_existing_destination(tmp_path):
    destination = tmp_path / "curves"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        export_empirical_curves(tmp_path, destination)

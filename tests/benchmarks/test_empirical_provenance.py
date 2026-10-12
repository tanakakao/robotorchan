"""Tests for tamper-evident portable empirical evidence manifests."""

import pytest

from robotorchan.benchmarks.empirical_provenance import (
    create_evidence_manifest,
    verify_evidence_manifest,
)


def test_evidence_provenance_roundtrip(tmp_path):
    (tmp_path / "metrics.csv").write_text("seed,regret\n0,1.0\n", encoding="utf-8")
    manifest = create_evidence_manifest(
        tmp_path,
        git_sha="a" * 40,
        experiment_config={"seed": 0, "q": 1},
    )
    assert manifest["files"][0]["path"] == "metrics.csv"
    assert verify_evidence_manifest(tmp_path) == manifest


def test_evidence_provenance_detects_modified_artifact(tmp_path):
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("a", encoding="utf-8")
    create_evidence_manifest(tmp_path, git_sha="a" * 40, experiment_config={})
    artifact.write_text("b", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_evidence_manifest(tmp_path)


def test_evidence_provenance_detects_extra_artifact(tmp_path):
    create_evidence_manifest(tmp_path, git_sha="a" * 40, experiment_config={})
    (tmp_path / "extra.csv").write_text("new", encoding="utf-8")
    with pytest.raises(ValueError, match="membership"):
        verify_evidence_manifest(tmp_path)


def test_evidence_provenance_rejects_overwrite(tmp_path):
    create_evidence_manifest(tmp_path, git_sha="a" * 40, experiment_config={})
    with pytest.raises(FileExistsError):
        create_evidence_manifest(tmp_path, git_sha="a" * 40, experiment_config={})

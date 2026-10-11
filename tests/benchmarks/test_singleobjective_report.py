"""Validate Phase 42 evidence reporting gates."""

import json

import pytest

from robotorchan.benchmarks.singleobjective_report import summarize_singleobjective_evidence


def test_rejects_incomplete_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "failed", "problem": "branin", "q": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete"):
        summarize_singleobjective_evidence(tmp_path)
    assert not (tmp_path / "summary.csv").exists()


def test_rejects_missing_seed_trajectories(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "runtime_validated",
                "problem": "branin",
                "q": 1,
                "seeds": list(range(10)),
                "initial_points": 20,
                "evaluation_budget": 50,
                "dtype": "float64",
                "device": "cpu",
                "dirty": False,
                "strategies": ["random", "sobol"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing or unexpected"):
        summarize_singleobjective_evidence(tmp_path)
    assert not (tmp_path / "summary.csv").exists()

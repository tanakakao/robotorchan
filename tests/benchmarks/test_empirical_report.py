"""Guard against presenting incomplete empirical artifacts as measurements."""

import json

import pytest

from robotorchan.benchmarks.empirical_report import (
    save_empirical_report,
    summarize_empirical_directory,
)


def test_missing_manifest_is_not_a_completed_experiment(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize_empirical_directory(tmp_path)


def test_failed_manifest_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "failed", "problem": "branin"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="runtime-validated"):
        summarize_empirical_directory(tmp_path)


def test_incomplete_seed_artifacts_are_rejected(tmp_path):
    manifest = {
        "status": "runtime_validated",
        "problem": "branin",
        "strategies": ["random", "sobol", "qEI", "qNEI"],
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "q": 1,
        "dtype": "float64",
        "device": "cpu",
        "git_sha": "abc123",
        "pairing": {"design": True, "truth": True, "observation": True},
        "observation_noise": "problem_default_noiseless",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        summarize_empirical_directory(tmp_path)


def test_report_refuses_overwrite(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_empirical_report(tmp_path, output)

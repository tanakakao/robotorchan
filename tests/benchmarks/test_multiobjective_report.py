"""Check Phase 43 multiobjective evidence rejection gates."""

import json

import pytest

from robotorchan.benchmarks.multiobjective_report import summarize_multiobjective_evidence


def _manifest():
    return {
        "status": "runtime_validated",
        "problem": "branin_currin",
        "q": 3,
        "strategies": ["random", "sobol", "qEHVI", "qNEHVI"],
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "dtype": "float64",
        "device": "cpu",
        "dirty": False,
        "reference_point": [-310.0, -1.0],
        "objective_directions": ["maximize", "maximize"],
    }


@pytest.mark.parametrize("field,value", [("status", "failed"), ("q", 2), ("dirty", True)])
def test_rejects_nonconforming_evidence(tmp_path, field, value):
    manifest = _manifest()
    manifest[field] = value
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="nonconforming"):
        summarize_multiobjective_evidence(tmp_path)
    assert not (tmp_path / "summary.csv").exists()


def test_rejects_missing_seed_files(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing or unexpected"):
        summarize_multiobjective_evidence(tmp_path)
    assert not (tmp_path / "summary.csv").exists()

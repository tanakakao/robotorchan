"""Tests for fail-closed empirical benchmark evidence acceptance."""

from robotorchan.benchmarks.empirical_evidence_gate import audit_empirical_evidence

_FAMILIES = (
    "single_objective",
    "multi_objective",
    "constrained",
    "mixed_variable",
    "high_dimensional",
    "batch_async",
    "heterogeneous",
    "active_learning",
)


def _records():
    return [
        {
            "family": family,
            "execution": "measured",
            "completed_seeds": 10,
            "baseline": "random",
            "challenger": "sobol",
            "artifact_verified": True,
        }
        for family in _FAMILIES
    ]


def test_evidence_gate_accepts_complete_verified_inputs():
    report = audit_empirical_evidence(
        _records(), ci_success=True, provenance_verified=True, rerun_verified=True
    )
    assert report["accepted"]
    assert not report["failures"]


def test_evidence_gate_rejects_unmeasured_family():
    records = _records()
    records[0]["execution"] = "planned"
    report = audit_empirical_evidence(
        records, ci_success=True, provenance_verified=True, rerun_verified=True
    )
    assert not report["accepted"]
    assert "single_objective:not_measured" in report["failures"]


def test_evidence_gate_rejects_missing_family_and_rerun():
    report = audit_empirical_evidence(
        _records()[:-1], ci_success=True, provenance_verified=True, rerun_verified=False
    )
    assert not report["accepted"]
    assert "rerun:unverified" in report["failures"]
    assert "active_learning:missing" in report["failures"]

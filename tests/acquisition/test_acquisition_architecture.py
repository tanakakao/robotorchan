from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_acquisition_architecture_documents_botorch_first_contract() -> None:
    path = REPO_ROOT / "docs" / "development" / "acquisition-architecture.md"
    text = path.read_text(encoding="utf-8")

    assert "BoTorch-first" in text
    assert "not re-exported" in text
    assert "metadata, not a string-based factory" in text
    assert "must not create robotorchan wrapper" in text
    assert "Classification active-learning criteria are robotorchan-owned" in text
    assert "Heterogeneous semantic composition" in text
    assert "sample-residual feasibility" in text
    assert "posterior-predictive" in text
    assert "probability-residual feasibility" in text
    assert "FeasibilityWeightedAcquisition" in text
    assert "CandidateConstraints" in text
    assert "no shared `posterior()` contract" in text
    assert "entry-local at the BoTorch boundary" in text
    assert "sample_class_probabilities()" in text
    assert "must not be treated as sample-wise feasible" in text
    assert "resolve_acquisition_composition()" in text
    assert "AcquisitionCompositionPlan" in text
    assert "descriptive rather than executable" in text

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
    assert "must not be silently" in text
    assert "converted into one another" in text
    assert "FeasibilityWeightedAcquisition" in text
    assert "CandidateConstraints" in text

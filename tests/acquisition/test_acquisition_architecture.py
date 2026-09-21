from pathlib import Path


def test_acquisition_architecture_documents_botorch_first_contract() -> None:
    text = Path("docs/development/acquisition-architecture.md").read_text(encoding="utf-8")

    assert "BoTorch-first" in text
    assert "not re-exported" in text
    assert "No acquisition registry" in text
    assert "classification-specific" in text
    assert "outside the current scope" in text

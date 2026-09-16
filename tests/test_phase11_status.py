from pathlib import Path


def test_phase11_status_is_complete():
    text = (Path(__file__).parents[1] / "docs" / "phase11_status.txt").read_text(encoding="utf-8")
    assert text.strip() == "Phase 11: complete"

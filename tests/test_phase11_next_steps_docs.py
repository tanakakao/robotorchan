from pathlib import Path


def test_post_phase11_extensions_are_documented_separately():
    text = (Path(__file__).parents[1] / "docs" / "phase11_next_steps.md").read_text(encoding="utf-8")

    assert "optimize_acqf" in text
    assert "BAxUS" in text
    assert "TuRBO" in text

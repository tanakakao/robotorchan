from pathlib import Path


def test_phase11_summary_mentions_common_training_loss():
    text = (Path(__file__).parents[1] / "docs" / "phase11_summary.md").read_text(encoding="utf-8")
    assert "training_loss()" in text

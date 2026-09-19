from pathlib import Path


def test_training_api_note_distinguishes_mll_and_joint_loss():
    path = Path(__file__).parents[1] / "docs" / "development" / "training_api.md"
    text = path.read_text(encoding="utf-8")

    assert "make_mll()" in text
    assert "training_loss()" in text

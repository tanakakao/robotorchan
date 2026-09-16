from pathlib import Path


def test_joint_training_example_uses_common_loss():
    text = (Path(__file__).parents[1] / "docs" / "joint_training_example.py.txt").read_text(
        encoding="utf-8"
    )
    assert "model.training_loss()" in text

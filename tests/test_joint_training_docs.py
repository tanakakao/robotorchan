from pathlib import Path


def test_joint_training_documentation_exists_and_uses_common_contract():
    path = Path(__file__).parents[1] / "docs" / "joint_neural_training.md"
    text = path.read_text(encoding="utf-8")

    assert "training_loss()" in text
    assert "HybridAutoEncoderGP" in text
    assert "JointVAEGP" in text

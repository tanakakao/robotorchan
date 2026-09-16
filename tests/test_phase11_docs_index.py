from pathlib import Path


def test_phase11_documentation_index_links_core_guides():
    text = (Path(__file__).parents[1] / "docs" / "README_phase11.md").read_text(encoding="utf-8")

    assert "high_dimensional_inputs.md" in text
    assert "high_dimensional_model_selection.md" in text
    assert "joint_neural_training.md" in text

from pathlib import Path


def test_phase11_completion_documentation_tracks_finished_benchmark_stack():
    path = Path(__file__).parents[1] / "docs" / "phase11_completion.md"
    text = path.read_text(encoding="utf-8")

    assert "repeated-seed" in text
    assert "Sequential BO" in text
    assert "training_loss()" in text

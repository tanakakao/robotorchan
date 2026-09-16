from pathlib import Path


def test_phase11_done_marker():
    assert (Path(__file__).parents[1] / "docs" / "phase11_done").read_text().strip() == "complete"

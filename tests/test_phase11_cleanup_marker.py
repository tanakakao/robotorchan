from pathlib import Path


def test_phase11_cleanup_marker_exists():
    marker = Path(__file__).parents[1] / "docs" / ".phase11_cleanup_marker"
    assert marker.exists()

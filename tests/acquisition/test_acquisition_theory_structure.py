"""Regression tests for acquisition theory chapter integrity."""

from pathlib import Path

from scripts.check_acquisition_theory_structure import check_theory_structure

ROOT = Path(__file__).resolve().parents[2]


def test_acquisition_theory_has_no_duplicate_sections() -> None:
    """Theory chapters must not contain repeated sections or injected titles."""
    assert check_theory_structure(ROOT) == []


def test_acquisition_theory_detects_duplicate_sections(tmp_path: Path) -> None:
    """Duplicated numbered headings must fail structural validation."""
    chapter = tmp_path / "docs" / "theory" / "acquisition"
    chapter.mkdir(parents=True)
    (chapter / "05_information_theoretic.md").write_text(
        "# 5. Example\n\n## 5.1 First\n\n## 5.1 Again\n",
        encoding="utf-8",
    )
    failures = check_theory_structure(tmp_path)
    assert any("duplicate section 5.1" in failure for failure in failures)


def test_acquisition_theory_detects_injected_titles(tmp_path: Path) -> None:
    """A chapter title accidentally inserted into prose must fail."""
    chapter = tmp_path / "docs" / "theory" / "acquisition"
    chapter.mkdir(parents=True)
    (chapter / "09_active_learning.md").write_text(
        "# 9. Example\n\n## 9.1 First\n"
        "some prose # 9. Example\n",
        encoding="utf-8",
    )
    failures = check_theory_structure(tmp_path)
    assert any("embedded chapter title" in failure for failure in failures)

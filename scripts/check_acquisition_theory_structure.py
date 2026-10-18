"""Check acquisition theory chapters for duplicated or injected headings."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THEORY_ROOT = ROOT / "docs" / "theory" / "acquisition"
NUMBERED_HEADING_RE = re.compile(r"^## (\d+)\.(\d+)\s+(.+)$")
CHAPTER_TITLE_RE = re.compile(r"^# (\d+)\.\s+(.+)$")
INJECTED_TITLE_RE = re.compile(r"\S\s+# \d+\.\s+")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def check_chapter(path: Path) -> list[str]:
    """Report structural problems in one numbered theory chapter."""
    failures: list[str] = []
    chapter = int(path.name.split("_", 1)[0])
    content = path.read_text(encoding="utf-8")
    headings: set[int] = set()
    title_count = 0
    fence_char = ""
    fence_size = 0
    display_math_open = False

    for line_number, line in enumerate(content.splitlines(), 1):
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_size = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_size:
                fence_char, fence_size = "", 0
            continue
        if fence_char:
            continue
        if line.strip() == "$$":
            display_math_open = not display_math_open
            continue
        if display_math_open:
            continue

        title = CHAPTER_TITLE_RE.match(line)
        if title:
            title_count += 1
            if int(title.group(1)) != chapter:
                failures.append(f"{path.name}:{line_number}: wrong chapter title")
        if INJECTED_TITLE_RE.search(line):
            failures.append(f"{path.name}:{line_number}: embedded chapter title")
        match = NUMBERED_HEADING_RE.match(line)
        if match:
            heading_chapter, section = int(match.group(1)), int(match.group(2))
            if heading_chapter != chapter:
                failures.append(f"{path.name}:{line_number}: wrong section chapter")
            if section in headings:
                failures.append(f"{path.name}:{line_number}: duplicate section {chapter}.{section}")
            headings.add(section)

    if title_count != 1:
        failures.append(f"{path.name}: expected one chapter title, got {title_count}")
    if not headings:
        failures.append(f"{path.name}: missing numbered sections")
    if fence_char:
        failures.append(f"{path.name}: unclosed code fence")
    if display_math_open:
        failures.append(f"{path.name}: unclosed display math")
    return failures


def check_theory_structure(root: Path) -> list[str]:
    """Validate all numbered acquisition theory chapters."""
    theory = root / "docs" / "theory" / "acquisition"
    failures: list[str] = []
    for path in sorted(theory.glob("[0-9][0-9]_*.md")):
        failures.extend(check_chapter(path))
    return failures


def main() -> int:
    """Print findings and return a failing status when needed."""
    failures = check_theory_structure(ROOT)
    for failure in failures:
        print(failure)
    if failures:
        print(f"Found {len(failures)} acquisition theory structural errors.", file=sys.stderr)
        return 1
    print("Acquisition theory chapter structure is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

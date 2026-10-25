"""Check the stable theory–implementation mapping navigation contract."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "docs" / "development" / "theory-implementation"
FILES = ("README.md", "models.md", "acquisition.md", "optimization.md", "verification.md")
SUBJECTS = ("models.md", "acquisition.md", "optimization.md")


def check_mapping_consistency(root: Path = ROOT) -> list[str]:
    """Check mapping files and bidirectional navigation without interpreting theory."""
    mapping = root / "docs" / "development" / "theory-implementation"
    failures: list[str] = []
    for name in FILES:
        if not (mapping / name).is_file():
            failures.append(f"missing mapping document: {name}")
    if failures:
        return failures

    index = (mapping / "README.md").read_text(encoding="utf-8")
    evidence = (mapping / "verification.md").read_text(encoding="utf-8")
    for name in SUBJECTS:
        if not re.search(r"\]\(" + re.escape(name) + r"\)", index):
            failures.append(f"index missing subject link: {name}")
        if not re.search(r"\]\(" + re.escape(name) + r"\)", evidence):
            failures.append(f"verification missing subject link: {name}")
        content = (mapping / name).read_text(encoding="utf-8")
        for target in ("README.md", "verification.md", "../../theory/README.md"):
            if not re.search(r"\]\(" + re.escape(target) + r"\)", content):
                failures.append(f"{name} missing navigation link: {target}")
    if not re.search(r"\]\(../../theory/README\.md\)", index):
        failures.append("index missing canonical theory link")
    return failures


def main() -> int:
    """Exit unsuccessfully when required navigation is absent."""
    failures = check_mapping_consistency()
    for failure in failures:
        print(failure)
    if failures:
        return 1
    print("Theory–implementation mapping navigation is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

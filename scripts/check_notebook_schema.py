"""Validate Jupyter notebook schemas without executing cells."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".venv", "node_modules"}


def validate_notebooks(root: Path) -> tuple[int, list[str]]:
    """Validate discovered notebooks and return the number checked and errors."""
    count = 0
    failures = []
    for path in sorted(root.rglob("*.ipynb")):
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        count += 1
        try:
            notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
            nbformat.validate(notebook)
        except (OSError, ValueError, nbformat.ValidationError) as exc:
            failures.append(f"{path.relative_to(root)}: {exc}")
    return count, failures


def main() -> int:
    count, failures = validate_notebooks(ROOT)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        print(f"Invalid notebooks: {len(failures)}/{count}", file=sys.stderr)
        return 1
    print(f"Validated {count} notebook files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate every committed Jupyter notebook without executing its cells."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    notebooks = sorted(ROOT.rglob("*.ipynb"))
    failures = []
    for path in notebooks:
        if any(part in {".git", ".venv", "node_modules"} for part in path.parts):
            continue
        try:
            notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
            nbformat.validate(notebook)
        except (OSError, ValueError, nbformat.ValidationError) as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Validated {len(notebooks)} notebook files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

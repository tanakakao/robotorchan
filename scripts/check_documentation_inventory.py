"""Validate Markdown inventory coverage and retained migration destinations."""

from __future__ import annotations

import json
from pathlib import Path

from check_docs_links import ROOT, markdown_sources

INVENTORY = Path("docs-internal/maintenance/documentation-inventory.json")
ACTIONS = {"KEEP", "MOVE", "MERGE", "SPLIT", "ARCHIVE", "DELETE"}


def check_inventory(root: Path = ROOT) -> list[str]:
    """Reject missing routes and Markdown files without an ownership decision."""
    data = json.loads((root / INVENTORY).read_text(encoding="utf-8"))
    failures = []
    seen = set()
    classified = set()
    for row in data["files"]:
        source = row["current_path"]
        if source in seen:
            failures.append(f"duplicate baseline document: {source}")
        seen.add(source)
        if row["action"] not in ACTIONS:
            failures.append(f"unknown action: {source}")
        if not row.get("category") or not row.get("reason"):
            failures.append(f"missing ownership decision: {source}")
        destinations = [row["destination"], *row.get("extracted_destinations", [])]
        for destination in destinations:
            classified.add(destination)
            target = (root / destination).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                failures.append(f"missing or unsafe destination: {destination}")
    for row in data.get("generated_documents", []):
        classified.add(row["path"])
        if not (root / row["path"]).is_file():
            failures.append(f"missing generated document: {row['path']}")
    actual = {p.relative_to(root).as_posix() for p in markdown_sources(root)}
    failures.extend(f"unclassified Markdown: {path}" for path in sorted(actual - classified))
    return failures


def main() -> int:
    """Print actionable failures without claiming semantic review."""
    failures = check_inventory()
    for failure in failures:
        print(failure)
    if failures:
        return 1
    print("Markdown inventory destinations and classification coverage are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

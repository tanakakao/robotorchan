"""Check local links in Markdown cells of example notebooks."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from check_docs_links import strip_fenced_code

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\((<[^>]+>|[^)\s]+)(?:\s+['\"][^)]*['\"])?\)")


def check_notebook_links(root: Path) -> list[str]:
    """Return missing relative link targets from notebook Markdown cells."""
    failures = []
    for notebook in sorted((root / "examples" / "notebooks").glob("*.ipynb")):
        data = json.loads(notebook.read_text(encoding="utf-8"))
        for index, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") != "markdown":
                continue
            content = strip_fenced_code("".join(cell.get("source", [])))
            for match in LINK_RE.finditer(content):
                raw = match.group(1).strip("<>")
                parsed = urlsplit(raw)
                if parsed.scheme or parsed.netloc or raw.startswith(("#", "//")):
                    continue
                if not parsed.path:
                    continue
                target = (notebook.parent / unquote(parsed.path)).resolve()
                if not target.is_relative_to(root.resolve()):
                    failures.append(
                        f"{notebook.relative_to(root)} cell {index}: outside repository: {raw}"
                    )
                elif not target.exists():
                    failures.append(
                        f"{notebook.relative_to(root)} cell {index}: missing target: {raw}"
                    )
    return failures


def main() -> int:
    """Report invalid notebook links."""
    failures = check_notebook_links(ROOT)
    for failure in failures:
        print(failure)
    if failures:
        print(f"Found {len(failures)} broken notebook links.", file=sys.stderr)
        return 1
    print("Local notebook Markdown links are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

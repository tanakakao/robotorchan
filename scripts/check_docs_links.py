"""Validate local Markdown links under docs/ using only the standard library."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\((<[^>]+>|[^)\s]+)(?:\s+['\"][^)]*['\"])?\)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
ROOT = Path(__file__).resolve().parents[1]


def strip_fenced_code(content: str) -> str:
    """Exclude fenced code examples from Markdown link inspection."""
    lines = []
    fence_char = None
    fence_size = 0
    for line in content.splitlines():
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if fence_char is None:
                fence_char = marker[0]
                fence_size = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_size:
                fence_char = None
                fence_size = 0
            lines.append("")
        else:
            lines.append("" if fence_char is not None else line)
    return "\n".join(lines)


def check_links(root: Path) -> list[str]:
    """Return missing local targets for Markdown files under docs/."""
    failures = []
    for source in sorted((root / "docs").rglob("*.md")):
        content = strip_fenced_code(source.read_text(encoding="utf-8"))
        for match in LINK_RE.finditer(content):
            raw = match.group(1).strip("<>")
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc or raw.startswith(("#", "//")):
                continue
            if not parsed.path:
                continue
            target = (source.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(root.resolve()):
                failures.append(f"{source.relative_to(root)}: outside repository: {raw}")
            elif not target.exists():
                failures.append(f"{source.relative_to(root)}: missing target: {raw}")
    return failures


def main() -> int:
    """Print broken links and return nonzero on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    failures = check_links(args.root.resolve())
    for failure in failures:
        print(failure)
    if failures:
        print(f"Found {len(failures)} missing local Markdown link targets.", file=sys.stderr)
        return 1
    print("Local Markdown link targets under docs/ are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

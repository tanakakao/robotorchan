"""Validate local reference-style links and Markdown heading fragments."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from check_docs_links import LINK_RE, ROOT, strip_fenced_code

DEFINITION_RE = re.compile(r'^\s{0,3}\[([^\]]+)\]:\s*<?([^\s>]+)>?', re.MULTILINE)
REFERENCE_RE = re.compile(r'(?<!!)\[([^\]]+)\]\[([^\]]*)\]')
HEADING_RE = re.compile(r'^#{1,6}\s+(.+?)\s*#*\s*$', re.MULTILINE)
EXPLICIT_ID_RE = re.compile(r'\{#([\w-]+)\}\s*$')


def normalize(label: str) -> str:
    """Normalize a Markdown reference label."""
    return " ".join(label.casefold().split())


def anchors(text: str) -> set[str]:
    """Collect GitHub-style heading anchors, including duplicate suffixes."""
    result = set()
    counts = {}
    for match in HEADING_RE.finditer(strip_fenced_code(text)):
        heading = re.sub(r'<[^>]+>', '', match.group(1))
        explicit = EXPLICIT_ID_RE.search(heading)
        if explicit:
            result.add(explicit.group(1))
            heading = heading[:explicit.start()]
        heading = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', heading)
        heading = re.sub(r'[^\w\s-]', '', heading.casefold()).strip()
        slug = re.sub(r'\s+', '-', heading)
        if not slug:
            continue
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(f"{slug}-{count}" if count else slug)
    return result


def check_reference_anchors(root: Path) -> list[str]:
    """Report invalid local reference links and Markdown fragments."""
    failures = []
    documents = sorted((root / "docs").rglob("*.md"))
    for source in documents:
        content = strip_fenced_code(source.read_text(encoding="utf-8"))
        definitions = {
            normalize(label): url
            for label, url in DEFINITION_RE.findall(content)
        }
        links = [match.group(1).strip("<>") for match in LINK_RE.finditer(content)]
        for match in REFERENCE_RE.finditer(content):
            label = normalize(match.group(2) or match.group(1))
            if label not in definitions:
                failures.append(f"{source.relative_to(root)}: undefined reference: {label}")
            else:
                links.append(definitions[label])
        for url in links:
            parsed = urlsplit(url)
            if not parsed.fragment or parsed.scheme or parsed.netloc:
                continue
            target = (source.parent / unquote(parsed.path)).resolve() if parsed.path else source
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                continue  # Existing local-path checker owns missing targets.
            if target.suffix.lower() != ".md":
                continue
            fragment = unquote(parsed.fragment)
            if fragment not in anchors(target.read_text(encoding="utf-8")):
                failures.append(f"{source.relative_to(root)}: missing anchor: {url}")
    return failures


def main() -> int:
    """Run reference and fragment validation."""
    failures = check_reference_anchors(ROOT)
    for failure in failures:
        print(failure)
    if failures:
        print(f"Found {len(failures)} invalid references or anchors.", file=sys.stderr)
        return 1
    print("Markdown reference links and heading anchors are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

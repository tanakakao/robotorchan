"""Check Theory ownership and concrete Guide source references."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from check_docs_links import ROOT, strip_fenced_code

SOURCE_RE = re.compile(r"\bsrc/robotorchan/[\w./-]+\.py\b")
CLASS_RE = re.compile(r"(?m)^class (\w+)")
LITERATURE_NAMES = {
    "BALD",
    "CVaR",
    "Expectation",
    "GP",
    "Kernel",
    "MeanVariance",
    "Model",
    "Posterior",
    "RandomizedStraddle",
    "Straddle",
    "WorstCase",
}


def guide_usage_blocks(text: str) -> list[int]:
    """Find top-level workflow code while allowing implementation definitions."""
    failures = []
    pattern = r"(?m)^(?P<f>```|~~~)(?:python|py)\s*\n(.*?)^(?P=f)\s*$"
    for index, match in enumerate(re.finditer(pattern, text, re.DOTALL), 1):
        try:
            tree = ast.parse(match.group(2))
        except SyntaxError:
            # An actual source excerpt can omit surrounding class/function context.
            continue
        for node in tree.body:
            if isinstance(node, ast.Assert):
                failures.append(index)
                break
            targets = node.targets if isinstance(node, ast.Assign) else []
            if any(isinstance(target, ast.Name) and target.id == "model" for target in targets):
                failures.append(index)
                break
            calls = ast.walk(node) if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) else []
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr
                in {"fit", "predict", "predict_proba", "posterior", "plot", "show"}
                for call in calls
            ):
                failures.append(index)
                break
    return failures


def check_boundaries(root: Path = ROOT) -> list[str]:
    """Reject package instructions in Theory and broken source paths in Guide."""
    failures = []
    names = set()
    for source in (root / "src" / "robotorchan").rglob("*.py"):
        names.update(CLASS_RE.findall(source.read_text(encoding="utf-8")))
    names -= LITERATURE_NAMES
    for path in sorted((root / "docs" / "theory").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"robotorchan|\b(?:src|tests)/|```python", text):
            failures.append(f"implementation instructions in Theory: {path.relative_to(root)}")
        for name in sorted(names):
            if re.search(r"\b" + re.escape(name) + r"\b", text):
                failures.append(f"concrete class in Theory: {path.relative_to(root)}: {name}")
    for path in sorted((root / "docs" / "guide").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for reference in sorted(set(SOURCE_RE.findall(text))):
            if not (root / reference).is_file():
                failures.append(f"missing Guide source: {path.relative_to(root)}: {reference}")
        if re.search(r"\bPhase\s+\d+", strip_fenced_code(text)):
            failures.append(f"development-cycle record in Guide: {path.relative_to(root)}")
        for index in guide_usage_blocks(text):
            failures.append(
                f"usage or verification code in Guide: {path.relative_to(root)}: {index}"
            )
    return failures


def main() -> int:
    """Report ownership failures without certifying mathematical correctness."""
    failures = check_boundaries()
    for failure in failures:
        print(failure)
    if failures:
        return 1
    print("Theory ownership and Guide source references are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

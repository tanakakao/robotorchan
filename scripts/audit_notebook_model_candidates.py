"""Audit executable model-name evidence in the registered notebook examples."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "docs" / "model_coverage.json"
OUTPUT = ROOT / "docs-internal" / "maintenance" / "notebook-model-candidate-coverage.csv"


def inspect_notebook(path: Path) -> tuple[set[str], bool, bool]:
    """Return called constructor names and conservative candidate/evaluation signals."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    constructors: set[str] = set()
    imports: dict[str, str] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        tree = ast.parse("".join(source) if isinstance(source, list) else source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports[alias.asname or alias.name] = alias.name
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                constructors.add(imports.get(node.func.id, node.func.id))
            elif isinstance(node.func, ast.Attribute):
                constructors.add(node.func.attr)
    candidate = any(
        token in code
        for token in ("optimize_acqf(", "optimize_acqf_mixed(", "candidate_pool", "candidate =")
    )
    evaluation = any(
        token in code for token in ("observed_value", "objective(candidate)", "new_y", "new_Y")
    )
    return constructors, candidate, evaluation


def build_rows() -> list[dict[str, str]]:
    """Cross-reference registry mapping with actual constructor calls in code cells."""
    registry = json.loads(COVERAGE.read_text(encoding="utf-8"))["models"]
    cache: dict[str, tuple[set[str], bool, bool]] = {}
    rows = []
    for name, metadata in sorted(registry.items()):
        notebook = metadata["notebook"]
        if notebook not in cache:
            cache[notebook] = inspect_notebook(ROOT / notebook)
        constructors, candidate, evaluation = cache[notebook]
        called = name in constructors
        if not called:
            status = "unverified"
        elif candidate and evaluation:
            status = "static_pipeline"
        else:
            status = "partial"
        rows.append(
            {
                "model": name,
                "notebook": notebook,
                "constructor_called": str(called).lower(),
                "candidate_signal": str(candidate).lower(),
                "evaluation_signal": str(evaluation).lower(),
                "status": status,
            }
        )
    return rows


def render() -> str:
    """Generate a deterministic CSV without claiming runtime model coverage."""
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "model",
            "notebook",
            "constructor_called",
            "candidate_signal",
            "evaluation_signal",
            "status",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(build_rows())
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = render()
    if args.check:
        rows = build_rows()
        counts = {
            status: sum(row["status"] == status for row in rows)
            for status in ("static_pipeline", "partial", "unverified")
        }
        print(f"Model candidate static audit: {counts}")
        print("Unverified model names (mapped notebook):")
        for row in rows:
            if row["status"] == "unverified":
                print(f"  {row['model']}: {row['notebook']}")
        if not rows:
            return 1
    else:
        OUTPUT.write_text(result, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

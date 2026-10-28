"""Report static training and candidate-generation evidence for each notebook.

Evidence flags are not execution results and must not be used as pass/fail claims.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "examples/notebooks"
FIELDS = [
    "notebook",
    "parseable_code_cells",
    "invalid_code_cells",
    "model_fit_calls",
    "native_fit_calls",
    "posterior_calls",
    "acquisition_mentions",
    "optimizer_calls",
    "candidate_mentions",
    "execution",
]


def calls_in(source: str) -> set[str]:
    tree = ast.parse(source)
    calls = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    return calls


def audit(path: Path) -> dict[str, str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    parsed = 0
    invalid = 0
    calls = set()
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = "".join(source) if isinstance(source, list) else source
        sources.append(source)
        try:
            calls.update(calls_in(source))
            parsed += 1
        except SyntaxError:
            invalid += 1
    text = "\n".join(sources)
    return {
        "notebook": path.name,
        "parseable_code_cells": str(parsed),
        "invalid_code_cells": str(invalid),
        "model_fit_calls": str("fit" in calls).lower(),
        "native_fit_calls": str(
            bool(calls & {"fit_gpytorch_mll", "fit_fully_bayesian_model_nuts"})
        ).lower(),
        "posterior_calls": str("posterior" in calls).lower(),
        "acquisition_mentions": str(
            any(token in text for token in ("Acquisition", "ExpectedImprovement", "qLog"))
        ).lower(),
        "optimizer_calls": str(
            bool(calls & {"optimize_acqf", "optimize_acqf_mixed", "optimize_acqf_discrete"})
        ).lower(),
        "candidate_mentions": str("candidate" in text.lower()).lower(),
        "execution": "unverified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notebook-pipeline-evidence.csv",
    )
    args = parser.parse_args()
    rows = [audit(path) for path in sorted(NOTEBOOK_DIR.glob("*.ipynb"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Notebooks inspected: {len(rows)}")
    print(f"Notebooks with invalid Python code cells: {sum(int(r['invalid_code_cells']) > 0 for r in rows)}")
    print("All execution statuses remain unverified.")


if __name__ == "__main__":
    main()

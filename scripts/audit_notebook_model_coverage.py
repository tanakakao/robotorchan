"""Audit public-model mentions and notebook pipeline evidence without executing notebooks.

The output is a triage inventory, not a runtime compatibility certification.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_INIT = ROOT / "src/robotorchan/models/__init__.py"
NOTEBOOK_DIR = ROOT / "examples/notebooks"
NON_MODELS = {
    "DocumentationLinks",
    "HighDimensionalStrategy",
    "InferenceType",
    "InputType",
    "ModelCapabilities",
    "ModelRegistryEntry",
    "PosteriorSamplingType",
    "RobustnessType",
    "TaskType",
    "UnsupportedModelOperationError",
    "get_model_registry_entry",
}
FIELDS = [
    "model",
    "notebooks",
    "code_references",
    "constructor_calls",
    "fit_api",
    "fit_method",
    "acquisition",
    "optimizer",
    "candidate",
    "execution",
]


def public_models() -> list[str]:
    tree = ast.parse(MODELS_INIT.read_text(encoding="utf-8"))
    exports = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            exports = ast.literal_eval(node.value)
    if not exports:
        raise ValueError("No public exports found in models/__init__.py")
    return sorted(set(exports) - NON_MODELS)


def notebook_cells() -> dict[str, list[dict]]:
    result = {}
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        data = json.loads(path.read_text(encoding="utf-8"))
        result[path.name] = data.get("cells", [])
    return result


def cell_text(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def code_evidence(cells: list[dict], model: str) -> tuple[bool, bool]:
    referenced = False
    constructed = False
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell_text(cell)
        if re.search(rf"\b{re.escape(model)}\b", source):
            referenced = True
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = func.id if isinstance(func, ast.Name) else None
            if isinstance(func, ast.Attribute):
                called = func.attr
            if called == model:
                constructed = True
    return referenced, constructed


def inventory() -> list[dict[str, str]]:
    notebooks = notebook_cells()
    rows = []
    for model in public_models():
        pattern = re.compile(rf"\b{re.escape(model)}\b")
        mentions = []
        code_refs = []
        constructors = []
        for name, cells in notebooks.items():
            if any(pattern.search(cell_text(cell)) for cell in cells):
                mentions.append(name)
            referenced, constructed = code_evidence(cells, model)
            if referenced:
                code_refs.append(name)
            if constructed:
                constructors.append(name)
        rows.append(
            {
                "model": model,
                "notebooks": ";".join(mentions),
                "code_references": ";".join(code_refs),
                "constructor_calls": ";".join(constructors),
                "fit_api": "unverified",
                "fit_method": "unverified",
                "acquisition": "unverified",
                "optimizer": "unverified",
                "candidate": "unverified",
                "execution": "unverified",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "notebook-model-inventory.csv")
    args = parser.parse_args()
    rows = inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"public exports: {len(rows)}")
    print(f"no notebook mention: {sum(not row['notebooks'] for row in rows)}")
    print(f"no code reference: {sum(not row['code_references'] for row in rows)}")
    print(f"no syntactic constructor call: {sum(not row['constructor_calls'] for row in rows)}")
    print("Constructor syntax does not prove execution or a successful candidate proposal.")


if __name__ == "__main__":
    main()

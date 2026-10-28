"""Generate a conservative static inventory of public models and notebooks.

Static inspection cannot establish runtime fit support or successful candidate generation.
Those fields remain unverified until checked against the actual API and execution.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_INIT = ROOT / "src/robotorchan/models/__init__.py"
NOTEBOOK_DIR = ROOT / "examples/notebooks"
FIELDS = [
    "model",
    "notebooks",
    "fit_api",
    "fit_method",
    "acquisition",
    "optimizer",
    "candidate",
    "execution",
]


def public_models() -> list[str]:
    tree = ast.parse(MODELS_INIT.read_text(encoding="utf-8"))
    exports: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            exports = ast.literal_eval(node.value)
    excluded = {
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
    return sorted(set(exports) - excluded)


def notebook_sources() -> dict[str, str]:
    result = {}
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        data = json.loads(path.read_text(encoding="utf-8"))
        result[path.name] = "\n".join(
            "".join(cell.get("source", [])) for cell in data.get("cells", [])
        )
    return result


def inventory() -> list[dict[str, str]]:
    sources = notebook_sources()
    rows = []
    for model in public_models():
        matches = [name for name, source in sources.items() if model in source]
        rows.append(
            {
                "model": model,
                "notebooks": ";".join(matches),
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
    uncovered = sum(not row["notebooks"] for row in rows)
    print(f"public models: {len(rows)}; no textual notebook match: {uncovered}")
    print("Textual matches are not proof of model execution or coverage.")


if __name__ == "__main__":
    main()

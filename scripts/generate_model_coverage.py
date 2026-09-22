"""Generate the documentation coverage artifact from the model registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robotorchan.models.registry import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "model_coverage.json"
EXCLUDED_PUBLIC_EXPORTS = ("UnsupportedModelOperationError",)
COVERAGE_CONTRACT = {
    "guide": "model-family guide that names or covers the public model family",
    "theory": "statistical or optimization theory shared by the model",
    "notebook": "representative runnable example; one notebook may cover multiple variants",
}


def build_model_coverage() -> dict[str, Any]:
    """Build serializable documentation coverage from MODEL_REGISTRY."""
    models = {}
    for name in sorted(MODEL_REGISTRY):
        docs = MODEL_REGISTRY[name].documentation
        models[name] = {
            "guide": docs.guide,
            "theory": docs.theory,
            "notebook": docs.notebook,
        }
    return {
        "schema_version": 3,
        "source_of_truth": "robotorchan.models.registry.MODEL_REGISTRY",
        "excluded_public_exports": list(EXCLUDED_PUBLIC_EXPORTS),
        "models": models,
        "coverage_contract": COVERAGE_CONTRACT,
    }


def render_model_coverage() -> str:
    """Render the generated artifact with stable formatting."""
    return json.dumps(build_model_coverage(), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    """Write the artifact or fail when the committed copy is stale."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_model_coverage()
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != rendered:
            print("docs/model_coverage.json is stale; regenerate it.")
            return 1
        return 0
    OUTPUT.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

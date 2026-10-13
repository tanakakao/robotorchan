"""Inspect public model API signatures without constructing or fitting models.

Run: python scripts/audit_model_api.py [--output path/to/report.json]
Reports are temporary review artifacts; do not commit point-in-time snapshots.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from robotorchan import models
from robotorchan.models.base import FittableModelMixin
from robotorchan.models.classification.registry import CLASSIFICATION_MODEL_REGISTRY
from robotorchan.models.registry import MODEL_REGISTRY

METHODS = (
    "fit",
    "make_mll",
    "training_loss",
    "posterior",
    "condition_on_observations",
    "fantasize",
    "predict_proba",
    "predict_class",
    "sample_class_probabilities",
)
PROPERTIES = ("raw_train_X", "raw_train_Y", "raw_train_Yvar")


def _signature(value: Any) -> str | None:
    """Return a signature when Python introspection supports the callable."""
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def _method_info(model_class: type, name: str) -> dict[str, Any]:
    """Describe method presence and its defining class, not runtime support."""
    owner = next((cls for cls in model_class.__mro__ if name in cls.__dict__), None)
    if owner is None:
        return {"status": "absent"}
    value = getattr(model_class, name, None)
    if not callable(value):
        return {"status": "not_callable", "owner": owner.__name__}
    status = "declared"
    if name == "fit" and owner is FittableModelMixin:
        status = "placeholder"
    return {
        "status": status,
        "owner": f"{owner.__module__}.{owner.__qualname__}",
        "signature": _signature(value),
        "abstract": bool(getattr(value, "__isabstractmethod__", False)),
    }


def _property_info(model_class: type, name: str) -> dict[str, str]:
    """Report declared property ownership without evaluating a descriptor."""
    owner = next((cls for cls in model_class.__mro__ if name in cls.__dict__), None)
    if owner is None:
        return {"status": "absent"}
    return {"status": "declared", "owner": f"{owner.__module__}.{owner.__qualname__}"}


def _model_row(model_class: type, *, family: str, capabilities: Any) -> dict[str, Any]:
    """Create an auditable signature snapshot for one public model class."""
    return {
        "family": family,
        "class": f"{model_class.__module__}.{model_class.__qualname__}",
        "constructor": _signature(model_class),
        "supports_fit_flag": getattr(model_class, "supports_fit", None),
        "supports_mll_flag": getattr(model_class, "supports_mll", None),
        "inference": str(capabilities.inference),
        "methods": {name: _method_info(model_class, name) for name in METHODS},
        "properties": {name: _property_info(model_class, name) for name in PROPERTIES},
    }


def build_inventory() -> dict[str, Any]:
    """Build a deterministic inventory of regression and classification APIs."""
    regression = {
        name: _model_row(
            getattr(models, name),
            family="regression",
            capabilities=entry.capabilities,
        )
        for name, entry in sorted(MODEL_REGISTRY.items())
    }
    classification = {
        name: _model_row(
            entry.model_class,
            family="classification",
            capabilities=entry.capabilities,
        )
        for name, entry in sorted(CLASSIFICATION_MODEL_REGISTRY.items())
    }
    return {
        "schema_version": 1,
        "scope": "static_signature_inventory_not_runtime_capability_verification",
        "regression_models": regression,
        "classification_models": classification,
    }


def main() -> int:
    """Print an inventory or save it as a review-only JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a temporary JSON report")
    args = parser.parse_args()
    content = json.dumps(build_inventory(), ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(content, end="")
    else:
        args.output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

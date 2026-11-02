"""Inventory model, acquisition, and optimizer contracts without running optimization.

Run: python scripts/audit_acquisition_optimizer_matrix.py --output /tmp/bo-matrix.json
The report is static metadata, not evidence of runtime compatibility.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from robotorchan.acquisition.compatibility import (
    check_capabilities_acquisition_compatibility,
    check_model_acquisition_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models.classification.registry import CLASSIFICATION_MODEL_REGISTRY
from robotorchan.models.registry import MODEL_REGISTRY
from robotorchan.optim.capabilities import OPTIMIZER_CAPABILITIES


def _serialize(value: Any) -> Any:
    """Convert dataclass capability metadata into deterministic JSON values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted((_serialize(item) for item in value), key=str)
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def build_inventory() -> dict[str, Any]:
    """Collect registered capabilities without claiming runtime support."""
    models = {
        name: {
            "family": "regression",
            "capabilities": _serialize(asdict(entry.capabilities)),
        }
        for name, entry in sorted(MODEL_REGISTRY.items())
    }
    models.update(
        {
            name: {
                "family": "classification",
                "capabilities": _serialize(asdict(entry.capabilities)),
            }
            for name, entry in sorted(CLASSIFICATION_MODEL_REGISTRY.items())
        }
    )
    acquisitions = {
        name: {
            "capabilities": _serialize(asdict(entry.capabilities)),
            "implementation_strategy": entry.implementation_strategy,
        }
        for name, entry in sorted(ACQUISITION_REGISTRY.items())
    }
    optimizers = {
        name: _serialize(asdict(capabilities))
        for name, capabilities in sorted(OPTIMIZER_CAPABILITIES.items())
    }
    model_acquisition = {
        model_name: {
            acquisition_name: {
                "status": "static_compatible" if result.compatible else "static_incompatible",
                "reasons": list(result.reasons),
                "runtime_verified": False,
            }
            for acquisition_name in sorted(ACQUISITION_REGISTRY)
            for result in [check_model_acquisition_compatibility(model_name, acquisition_name)]
        }
        for model_name in sorted(MODEL_REGISTRY)
    }
    classification_acquisition = {
        model_name: {
            acquisition_name: {
                "status": "static_compatible" if result.compatible else "static_incompatible",
                "reasons": list(result.reasons),
                "runtime_verified": False,
            }
            for acquisition_name in sorted(ACQUISITION_REGISTRY)
            for result in [
                check_capabilities_acquisition_compatibility(entry.capabilities, acquisition_name)
            ]
        }
        for model_name, entry in sorted(CLASSIFICATION_MODEL_REGISTRY.items())
    }
    return {
        "schema_version": 3,
        "verification": "static_only",
        "models": dict(sorted(models.items())),
        "model_acquisition_static": model_acquisition,
        "classification_acquisition_static": classification_acquisition,
        "acquisitions": acquisitions,
        "optimizers": optimizers,
        "specialized_optimizer_paths": [
            "optimize_acqf_sequential",
            "optimize_mixed_one_shot_acqf",
        ],
        "limitations": [
            "Registry membership does not establish runtime compatibility.",
            "Specialized paths and composition factories require separate validation.",
            "No model-acquisition-optimizer triple is marked supported by this report.",
            "Classification pairs use capability metadata only, not runtime posterior checks.",
        ],
    }


def main() -> int:
    """Print or write the inventory JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(build_inventory(), ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        args.output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

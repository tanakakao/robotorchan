"""Inspect acquisition public APIs and registry metadata without constructing models.

Run: python scripts/audit_acquisition_api.py --output /tmp/acquisition-api.json

The JSON report is a point-in-time review artifact and must not be committed.
Static metadata and signatures are not evidence of runtime acquisition compatibility.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from botorch.acquisition.acquisition import AcquisitionFunction

import robotorchan.acquisition as acquisition
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY

PARAMETERS = (
    "model",
    "objective",
    "posterior_transform",
    "sampler",
    "X_pending",
    "constraints",
    "best_f",
    "ref_point",
    "partitioning",
    "num_fantasies",
)


def _signature(value: Any) -> str | None:
    """Return a Python signature when introspection is supported."""
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def _parameters(value: Any) -> list[str]:
    """Describe explicit parameters, excluding variable keyword arguments."""
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return []
    return [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        not in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    ]


def _entry_metadata(name: str) -> dict[str, Any] | None:
    """Keep acquisition capability claims separate from runtime verification."""
    entry = ACQUISITION_REGISTRY.get(name)
    if entry is None:
        return None
    capabilities = entry.capabilities
    return {
        "purpose": capabilities.purpose.value,
        "posterior_requirement": capabilities.posterior_requirement.value,
        "target": capabilities.target.value,
        "max_q": capabilities.max_q,
        "supports_multi_output": capabilities.supports_multi_output,
        "supports_ensemble": capabilities.supports_ensemble,
        "supports_constraints": capabilities.supports_constraints,
        "supports_multi_objective": capabilities.supports_multi_objective,
        "monte_carlo": capabilities.monte_carlo,
        "requires_fantasize": capabilities.requires_fantasize,
        "requires_multi_fidelity": capabilities.requires_multi_fidelity,
        "one_shot": capabilities.one_shot,
        "implementation_strategy": entry.implementation_strategy,
        "limitations": list(entry.limitations),
    }


def build_inventory() -> dict[str, Any]:
    """Audit local public symbols and upstream metadata-only acquisitions."""
    exports = {}
    local_acquisitions = set()
    for name in sorted(acquisition.__all__):
        value = getattr(acquisition, name)
        is_acquisition = (
            inspect.isclass(value)
            and issubclass(value, AcquisitionFunction)
            and value is not AcquisitionFunction
        )
        if is_acquisition:
            kind = "local_acquisition_class"
            local_acquisitions.add(name)
        elif inspect.isclass(value):
            kind = "supporting_class"
        elif callable(value):
            kind = "composition_or_support_helper"
        else:
            kind = "other_export"
        parameters = _parameters(value) if callable(value) else []
        exports[name] = {
            "kind": kind,
            "qualified_name": getattr(value, "__module__", ""),
            "signature": _signature(value) if callable(value) else None,
            "parameters": parameters,
            "common_parameters": sorted(set(parameters).intersection(PARAMETERS)),
            "registry": _entry_metadata(name),
        }

    native = {
        name: _entry_metadata(name)
        for name, entry in sorted(ACQUISITION_REGISTRY.items())
        if entry.implementation_strategy.startswith("BoTorch-native")
    }
    unregistered_local = sorted(local_acquisitions.difference(ACQUISITION_REGISTRY))
    registry_local_missing_exports = sorted(
        name
        for name, entry in ACQUISITION_REGISTRY.items()
        if not entry.implementation_strategy.startswith("BoTorch-native") and name not in exports
    )
    return {
        "schema_version": 1,
        "verification": "static_only",
        "exports": exports,
        "native_botorch_metadata": native,
        "gaps": {
            "unregistered_public_acquisition_classes": unregistered_local,
            "registered_local_acquisitions_not_exported": registry_local_missing_exports,
        },
    }


def main() -> int:
    """Write a temporary JSON file or print the inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(build_inventory(), ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        args.output.write_text(result, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

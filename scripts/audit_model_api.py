"""Inspect public model API signatures without constructing or fitting models.

Run: python scripts/audit_model_api.py [--output path/to/report.json]
Reports are temporary review artifacts; do not commit point-in-time snapshots.
"""

from __future__ import annotations

import argparse
import ast
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
    upstream_bases = [
        f"{base.__module__}.{base.__qualname__}"
        for base in model_class.__mro__[1:]
        if base.__module__.startswith(("botorch.", "gpytorch.", "sklearn."))
    ]
    return {
        "family": family,
        "implementation_module": model_class.__module__,
        "upstream_bases": upstream_bases,
        "provenance_verification": "static_mro_only",
        "class": f"{model_class.__module__}.{model_class.__qualname__}",
        "constructor": _signature(model_class),
        "supports_fit_flag": getattr(model_class, "supports_fit", None),
        "supports_mll_flag": getattr(model_class, "supports_mll", None),
        "inference": str(capabilities.inference),
        "methods": {name: _method_info(model_class, name) for name in METHODS},
        "properties": {name: _property_info(model_class, name) for name in PROPERTIES},
    }


def classify_training_api(row: dict[str, Any]) -> dict[str, str]:
    """Classify observed fitting interfaces without claiming runtime success."""
    fit = row["methods"]["fit"]["status"]
    loss = row["methods"]["training_loss"]["status"]
    mll = row["methods"]["make_mll"]["status"]
    inference = row["inference"]
    if fit == "declared":
        category = "fit_declared"
    elif fit == "placeholder" or row["supports_fit_flag"] is True:
        category = "fit_contract_needs_review"
    elif inference == "fully_bayesian":
        category = "specialized_mcmc_training"
    elif loss == "declared":
        category = "custom_loss_training"
    elif row["supports_mll_flag"] is True and mll == "declared":
        category = "mll_training_fit_candidate"
    else:
        category = "manual_training_review"
    return {
        "category": category,
        "verification": "static_only",
        "next_step": "verify_constructor_training_and_posterior_at_runtime",
    }


def find_contract_gaps(inventory: dict[str, Any]) -> list[dict[str, str]]:
    """Return explicit static mismatches for review rather than hiding them."""
    gaps = []
    for group in ("regression_models", "classification_models"):
        for name, row in inventory[group].items():
            fit = row["methods"]["fit"]["status"]
            if fit == "declared" and row["supports_fit_flag"] is not True:
                gaps.append(
                    {
                        "model": name,
                        "issue": "fit_declared_without_supports_fit_true",
                        "observed": str(row["supports_fit_flag"]),
                    }
                )
            if row["supports_fit_flag"] is True and fit != "declared":
                gaps.append(
                    {
                        "model": name,
                        "issue": "supports_fit_true_without_concrete_method",
                        "observed": fit,
                    }
                )
            if (
                row["supports_mll_flag"] is True
                and row["methods"]["make_mll"]["status"] != "declared"
            ):
                gaps.append(
                    {
                        "model": name,
                        "issue": "supports_mll_true_without_callable_method",
                        "observed": row["methods"]["make_mll"]["status"],
                    }
                )
            if group == "classification_models":
                for method in ("predict_proba", "predict_class"):
                    if row["methods"][method]["status"] != "declared":
                        gaps.append(
                            {
                                "model": name,
                                "issue": f"missing_{method}",
                                "observed": row["methods"][method]["status"],
                            }
                        )
    return gaps


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
    for name, row in regression.items():
        links = MODEL_REGISTRY[name].documentation
        row["documentation"] = {
            "guide": links.guide,
            "theory": links.theory,
            "notebook": links.notebook,
            "verification": "registry_metadata_only",
        }
    for row in classification.values():
        row["documentation"] = {
            "verification": "not_recorded_in_classification_registry",
        }
    for row in (*regression.values(), *classification.values()):
        row["training_api"] = classify_training_api(row)
    inventory = {
        "schema_version": 2,
        "scope": "static_signature_inventory_not_runtime_capability_verification",
        "regression_models": regression,
        "classification_models": classification,
    }
    inventory["static_contract_gaps"] = find_contract_gaps(inventory)
    return inventory


def find_missing_documentation_paths(
    inventory: dict[str, Any], *, repository_root: Path
) -> list[dict[str, str]]:
    """Check referenced files exist; do not claim notebook execution or theory validity."""
    missing = []
    for name, row in inventory["regression_models"].items():
        for kind in ("guide", "theory", "notebook"):
            reference = row["documentation"][kind]
            if not (repository_root / reference).is_file():
                missing.append({"model": name, "kind": kind, "path": reference})
    return missing


def find_test_references(
    inventory: dict[str, Any], *, repository_root: Path
) -> dict[str, dict[str, Any]]:
    """Find syntactic model references in tests; not evidence of model verification."""
    names = {
        name
        for group in ("regression_models", "classification_models")
        for name in inventory[group]
    }
    classes = {
        name: row["class"].rsplit(".", 1)[-1]
        for group in ("regression_models", "classification_models")
        for name, row in inventory[group].items()
    }
    matches: dict[str, set[str]] = {name: set() for name in names}
    tests_root = repository_root / "tests"
    for path in sorted(tests_root.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        referenced.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
        for name, class_name in classes.items():
            if class_name in referenced:
                matches[name].add(path.relative_to(repository_root).as_posix())
    return {
        name: {
            "test_files": sorted(matches[name]),
            "verification": "syntactic_reference_only",
        }
        for name in sorted(names)
    }


def find_upstream_constructor_calls(
    inventory: dict[str, Any], *, repository_root: Path
) -> dict[str, dict[str, Any]]:
    """Find explicit upstream constructor calls within model class syntax trees.

    Imported call targets are resolved statically. Calls via factories, aliases
    assigned at runtime, and helper functions are intentionally not inferred.
    """
    results: dict[str, dict[str, Any]] = {}
    for group in ("regression_models", "classification_models"):
        for name, row in inventory[group].items():
            module_name, class_name = row["class"].rsplit(".", 1)
            source = repository_root / "src" / Path(*module_name.split(".")).with_suffix(".py")
            calls: set[str] = set()
            if source.is_file():
                tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
                imports: dict[str, str] = {}
                import_nodes = (
                    child
                    for node in tree.body
                    if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    for child in ast.walk(node)
                )
                for node in import_nodes:
                    if isinstance(node, ast.ImportFrom) and node.module:
                        for alias in node.names:
                            imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            imports[alias.asname or alias.name.split(".")[0]] = alias.name
                for node in tree.body:
                    if not isinstance(node, ast.ClassDef) or node.name != class_name:
                        continue
                    for child in ast.walk(node):
                        if not isinstance(child, ast.Call):
                            continue
                        target = child.func
                        if isinstance(target, ast.Name):
                            resolved = imports.get(target.id, "")
                        elif isinstance(target, ast.Attribute) and isinstance(
                            target.value, ast.Name
                        ):
                            prefix = imports.get(target.value.id, "")
                            resolved = f"{prefix}.{target.attr}" if prefix else ""
                        else:
                            resolved = ""
                        if resolved.startswith(("botorch.", "gpytorch.", "sklearn.")):
                            calls.add(resolved)
            results[name] = {
                "upstream_calls": sorted(calls),
                "verification": "static_direct_import_call_only",
            }
    return results



def summarize_static_model_evidence(
    inventory: dict[str, Any],
    test_references: dict[str, dict[str, Any]],
    upstream_calls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize detected evidence, without inferring assertion or runtime coverage."""
    groups = {}
    for family in ("regression_models", "classification_models"):
        names = sorted(inventory[family])
        without_test_references = [
            name for name in names if not test_references[name]["test_files"]
        ]
        without_direct_upstream_calls = [
            name
            for name in names
            if not upstream_calls[name]["upstream_calls"]
        ]
        groups[family] = {
            "registered_models": len(names),
            "with_syntactic_test_references": len(names) - len(without_test_references),
            "without_syntactic_test_references": without_test_references,
            "with_direct_upstream_calls": len(names) - len(without_direct_upstream_calls),
            "without_direct_upstream_calls": without_direct_upstream_calls,
        }
    return {
        "verification": "static_detection_counts_only",
        "limitations": (
            "Missing matches do not imply missing tests or delegation; matches "
            "do not demonstrate executed assertions, runtime behavior or numerical validity."
        ),
        "families": groups,
    }


def main() -> int:
    """Print an inventory or save it as a review-only JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a temporary JSON report")
    parser.add_argument(
        "--include-test-references",
        action="store_true",
        help="Include syntactic test references, not verified test assertions",
    )
    parser.add_argument(
        "--include-upstream-calls",
        action="store_true",
        help="Include direct imported upstream calls within model class bodies",
    )
    parser.add_argument(
        "--include-evidence-summary",
        action="store_true",
        help="Summarize static model test references and direct upstream calls",
    )
    parser.add_argument(
        "--check-documentation-paths",
        action="store_true",
        help="Fail if regression registry guide/theory/notebook paths do not exist",
    )
    args = parser.parse_args()
    inventory = build_inventory()
    if args.include_test_references or args.include_evidence_summary:
        inventory["test_references"] = find_test_references(
            inventory, repository_root=Path(__file__).resolve().parents[1]
        )
    if args.include_upstream_calls or args.include_evidence_summary:
        inventory["upstream_calls"] = find_upstream_constructor_calls(
            inventory, repository_root=Path(__file__).resolve().parents[1]
        )
    if args.include_evidence_summary:
        inventory["evidence_summary"] = summarize_static_model_evidence(
            inventory,
            inventory["test_references"],
            inventory["upstream_calls"],
        )
    if args.check_documentation_paths:
        inventory["missing_documentation_paths"] = find_missing_documentation_paths(
            inventory, repository_root=Path(__file__).resolve().parents[1]
        )
    content = json.dumps(inventory, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(content, end="")
    else:
        args.output.write_text(content, encoding="utf-8")
    return int(bool(inventory.get("missing_documentation_paths")))


if __name__ == "__main__":
    raise SystemExit(main())

"""Inventory model, acquisition, and optimizer contracts without running optimization.

Run: python scripts/audit_acquisition_optimizer_matrix.py --output /tmp/bo-matrix.json
The report combines static metadata with explicitly scoped runtime test evidence.
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
from robotorchan.optim.capabilities import (
    BOTORCH_MIXED_OPTIMIZER_CAPABILITIES,
    MIXED_GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES,
    NSGA2_OPTIMIZER_CAPABILITIES,
    OPTIMIZER_CAPABILITIES,
)


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


# Runtime evidence is limited to the exact model, acquisition, backend, and q.
# CMA-ES tests use a mocked solver and are not end-to-end solver evidence.
_CLASSIFICATION_Q1_EVIDENCE = {
    "random": "tests/acquisition/test_classification_random_search_runtime.py",
    "ga": "tests/acquisition/test_classification_ga_runtime.py",
    "de": "tests/acquisition/test_classification_de_runtime.py",
    "pso": "tests/acquisition/test_classification_pso_runtime.py",
    "sobol": "tests/acquisition/test_classification_sobol_search_runtime.py",
}
_CLASSIFICATION_Q1_ACQUISITIONS = frozenset(
    {
        "BALD",
        "LatentStraddle",
        "MarginUncertainty",
        "PredictiveEntropy",
        "ProbabilityVariance",
    }
)


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
    specialized_optimizers = {
        "optimize_acqf_mixed_botorch": BOTORCH_MIXED_OPTIMIZER_CAPABILITIES,
        "optimize_acqf_mixed_ga": MIXED_GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES,
        "optimize_vector_nsga2": NSGA2_OPTIMIZER_CAPABILITIES,
    }
    all_optimizers = {
        **OPTIMIZER_CAPABILITIES,
        **specialized_optimizers,
    }
    acquisition_optimizer = {
        acquisition_name: {
            optimizer_name: {
                "status": (
                    "requires_specialized_path" if entry.capabilities.one_shot else "unverified"
                ),
                "reasons": (
                    ["One-shot acquisition requires a specialized optimization path."]
                    if entry.capabilities.one_shot
                    else []
                ),
                "optimizer_requires_grad": capabilities.requires_grad,
                "runtime_verified": False,
            }
            for optimizer_name, capabilities in sorted(all_optimizers.items())
        }
        for acquisition_name, entry in sorted(ACQUISITION_REGISTRY.items())
    }
    for acquisition_name in _CLASSIFICATION_Q1_ACQUISITIONS:
        for optimizer_name, test_path in _CLASSIFICATION_Q1_EVIDENCE.items():
            result = acquisition_optimizer[acquisition_name][optimizer_name]
            result["status"] = "verified_for_binary_single_task_gp_q1"
            result["runtime_verified"] = True
            result["evidence"] = [test_path]
            result["reasons"] = [
                "Verified only for BinarySingleTaskGPClassifier, q=1, unconstrained."
            ]
    for acquisition_name in _CLASSIFICATION_Q1_ACQUISITIONS:
        result = acquisition_optimizer[acquisition_name]["botorch"]
        result["status"] = "verified_for_binary_single_task_gp_q1"
        result["runtime_verified"] = True
        result["evidence"] = ["tests/acquisition/test_classification_acqf_optimizer_smoke.py"]
        result["reasons"] = ["Verified only for BinarySingleTaskGPClassifier, q=1, unconstrained."]
    for optimizer_name in ("torch_adam", "torch_adamw", "torch_sgd"):
        result = acquisition_optimizer["PredictiveEntropy"][optimizer_name]
        result["status"] = "verified_for_binary_single_task_gp_q1"
        result["runtime_verified"] = True
        result["evidence"] = ["tests/acquisition/test_classification_torch_optimizer_runtime.py"]
        result["reasons"] = ["Verified only for BinarySingleTaskGPClassifier, q=1, unconstrained."]
    hybrid_result = acquisition_optimizer["PredictiveEntropy"]["hybrid"]
    hybrid_result["status"] = "verified_for_binary_single_task_gp_q1"
    hybrid_result["runtime_verified"] = True
    hybrid_result["evidence"] = ["tests/acquisition/test_classification_hybrid_runtime.py"]
    hybrid_result["reasons"] = [
        "Verified only for BinarySingleTaskGPClassifier, q=1, unconstrained, DE global search."
    ]
    for acquisition_name in (
        "qLogExpectedImprovement",
        "qSimpleRegret",
        "qUpperConfidenceBound",
    ):
        for optimizer_name in ("botorch", "ga", "pso", "sobol"):
            result = acquisition_optimizer[acquisition_name][optimizer_name]
            result["status"] = "verified_for_single_task_gp_q1"
            result["runtime_verified"] = True
            result["evidence"] = [
                "tests/optim/integration/test_optimizer_model_acquisition_e2e.py"
            ]
            result["reasons"] = ["Verified only for SingleTaskGP, q=1, continuous, unconstrained."]
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
    verified_pairs = [
        (acquisition_name, optimizer_name)
        for acquisition_name, optimizer_results in acquisition_optimizer.items()
        for optimizer_name, result in optimizer_results.items()
        if result["runtime_verified"]
    ]
    unverified_pairs = [
        {
            "acquisition": acquisition_name,
            "optimizer": optimizer_name,
            "triage_group": (
                "specialized_path"
                if result["status"] == "requires_specialized_path"
                else "classification_existing_model"
                if acquisition_name in _CLASSIFICATION_Q1_ACQUISITIONS
                else "other_acquisitions"
            ),
        }
        for acquisition_name, optimizer_results in acquisition_optimizer.items()
        for optimizer_name, result in optimizer_results.items()
        if not result["runtime_verified"]
    ]
    total_pairs = sum(len(results) for results in acquisition_optimizer.values())
    return {
        "schema_version": 4,
        "verification": "static_inventory_with_scoped_runtime_evidence",
        "acquisition_optimizer_coverage": {
            "total_pairs": total_pairs,
            "scoped_runtime_verified_pairs": len(verified_pairs),
            "unverified_or_specialized_pairs": total_pairs - len(verified_pairs),
            "unverified_pairs": unverified_pairs,
            "verified_pairs": [
                {"acquisition": acquisition_name, "optimizer": optimizer_name}
                for acquisition_name, optimizer_name in verified_pairs
            ],
        },
        "external_botorch_acquisition_runtime_evidence": {
            acquisition_name: {
                optimizer_name: {
                    "status": "verified_for_single_task_gp_q1",
                    "runtime_verified": True,
                    "evidence": ["tests/optim/integration/test_optimizer_model_acquisition_e2e.py"],
                    "scope": "SingleTaskGP, q=1, unconstrained, continuous",
                }
                for optimizer_name in ("botorch", "ga", "pso", "sobol")
            }
            for acquisition_name in ("ExpectedImprovement", "UpperConfidenceBound")
        },
        "models": dict(sorted(models.items())),
        "model_acquisition_static": model_acquisition,
        "classification_acquisition_static": classification_acquisition,
        "acquisitions": acquisitions,
        "acquisition_optimizer_static": acquisition_optimizer,
        "optimizers": optimizers,
        "specialized_optimizers": {
            name: _serialize(asdict(capabilities))
            for name, capabilities in sorted(specialized_optimizers.items())
        },
        "specialized_optimizer_paths": [
            "optimize_acqf_sequential",
            "optimize_mixed_one_shot_acqf",
        ],
        "limitations": [
            "Registry membership does not establish runtime compatibility.",
            "Acquisition-optimizer evidence applies only to its stated model and test scope.",
            "One-shot paths require separate verification; q=1 does not remove this need.",
            "Specialized paths and composition factories require separate validation.",
            "Registry matrix does not mark model-acquisition-optimizer triples supported.",
            "External BoTorch acquisition evidence is separate from registered acquisitions.",
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

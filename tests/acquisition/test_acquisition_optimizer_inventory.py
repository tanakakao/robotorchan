"""Contract tests for the static three-layer inventory."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_acquisition_optimizer_matrix.py"
_SPEC = importlib.util.spec_from_file_location("audit_acquisition_optimizer_matrix", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_inventory = _MODULE.build_inventory


def test_inventory_covers_registered_public_metadata() -> None:
    from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
    from robotorchan.models.classification.registry import CLASSIFICATION_MODEL_REGISTRY
    from robotorchan.models.registry import MODEL_REGISTRY
    from robotorchan.optim.capabilities import OPTIMIZER_CAPABILITIES

    report = build_inventory()
    assert set(report["models"]) == set(MODEL_REGISTRY) | set(CLASSIFICATION_MODEL_REGISTRY)
    assert set(report["acquisitions"]) == set(ACQUISITION_REGISTRY)
    assert set(report["optimizers"]) == set(OPTIMIZER_CAPABILITIES)
    assert report["verification"] == "static_only"


def test_inventory_does_not_claim_runtime_compatibility() -> None:
    report = build_inventory()
    assert "compatibility" not in report
    assert "optimize_mixed_one_shot_acqf" in report["specialized_optimizer_paths"]


def test_static_model_acquisition_matrix_is_not_runtime_evidence() -> None:
    from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
    from robotorchan.models.registry import MODEL_REGISTRY

    report = build_inventory()
    matrix = report["model_acquisition_static"]
    assert set(matrix) == set(MODEL_REGISTRY)
    for acquisitions in matrix.values():
        assert set(acquisitions) == set(ACQUISITION_REGISTRY)
        for result in acquisitions.values():
            assert result["status"] in {"static_compatible", "static_incompatible"}
            assert result["runtime_verified"] is False
            assert isinstance(result["reasons"], list)


def test_classification_static_matrix_covers_registered_pairs() -> None:
    from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
    from robotorchan.models.classification.registry import CLASSIFICATION_MODEL_REGISTRY

    report = build_inventory()
    matrix = report["classification_acquisition_static"]
    assert set(matrix) == set(CLASSIFICATION_MODEL_REGISTRY)
    for acquisitions in matrix.values():
        assert set(acquisitions) == set(ACQUISITION_REGISTRY)
        for result in acquisitions.values():
            assert result["status"] in {"static_compatible", "static_incompatible"}
            assert result["runtime_verified"] is False
            assert isinstance(result["reasons"], list)


def test_inventory_covers_specialized_optimizer_capabilities() -> None:
    from dataclasses import asdict

    from robotorchan.optim.capabilities import (
        BOTORCH_MIXED_OPTIMIZER_CAPABILITIES,
        MIXED_GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES,
        NSGA2_OPTIMIZER_CAPABILITIES,
    )

    report = build_inventory()
    specialized = report["specialized_optimizers"]
    assert set(specialized) == {
        "optimize_acqf_mixed_botorch",
        "optimize_acqf_mixed_ga",
        "optimize_vector_nsga2",
    }
    assert specialized["optimize_acqf_mixed_botorch"]["mixed"] is True
    assert specialized["optimize_acqf_mixed_ga"]["categorical"] is True
    assert specialized["optimize_vector_nsga2"]["q_batch"] is False
    assert asdict(NSGA2_OPTIMIZER_CAPABILITIES)["q_batch"] is False
    assert BOTORCH_MIXED_OPTIMIZER_CAPABILITIES.mixed
    assert MIXED_GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES.mixed


def test_acquisition_optimizer_matrix_is_conservative_and_complete() -> None:
    from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
    from robotorchan.optim.capabilities import OPTIMIZER_CAPABILITIES

    report = build_inventory()
    matrix = report["acquisition_optimizer_static"]
    expected_optimizers = set(OPTIMIZER_CAPABILITIES) | set(report["specialized_optimizers"])
    assert set(matrix) == set(ACQUISITION_REGISTRY)
    for acquisition_name, optimizers in matrix.items():
        assert set(optimizers) == expected_optimizers
        for result in optimizers.values():
            assert result["status"] in {"unverified", "requires_specialized_path"}
            assert result["runtime_verified"] is False
            assert isinstance(result["optimizer_requires_grad"], bool)
            assert isinstance(result["reasons"], list)
        if ACQUISITION_REGISTRY[acquisition_name].capabilities.one_shot:
            assert all(
                result["status"] == "requires_specialized_path" for result in optimizers.values()
            )

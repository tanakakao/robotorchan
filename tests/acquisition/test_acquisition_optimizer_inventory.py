"""Contract tests for the static three-layer inventory."""

from scripts.audit_acquisition_optimizer_matrix import build_inventory


def test_inventory_covers_registered_public_metadata() -> None:
    from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
    from robotorchan.models.classification.registry import CLASSIFICATION_MODEL_REGISTRY
    from robotorchan.models.registry import MODEL_REGISTRY
    from robotorchan.optim.capabilities import OPTIMIZER_CAPABILITIES

    report = build_inventory()
    assert set(report["models"]) == set(MODEL_REGISTRY) | set(
        CLASSIFICATION_MODEL_REGISTRY
    )
    assert set(report["acquisitions"]) == set(ACQUISITION_REGISTRY)
    assert set(report["optimizers"]) == set(OPTIMIZER_CAPABILITIES)
    assert report["verification"] == "static_only"


def test_inventory_does_not_claim_runtime_compatibility() -> None:
    report = build_inventory()
    assert "compatibility" not in report
    assert "optimize_mixed_one_shot_acqf" in report["specialized_optimizer_paths"]

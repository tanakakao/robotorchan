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
    assert report["verification"] == "static_inventory_with_scoped_runtime_evidence"


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
            assert result["status"] in {
                "unverified",
                "requires_specialized_path",
                "verified_for_binary_single_task_gp_q1",
                "verified_for_single_task_gp_q1",
                "verified_for_model_list_gp_q1",
            }
            assert result["runtime_verified"] == (
                result["status"]
                in {
                    "verified_for_binary_single_task_gp_q1",
                    "verified_for_model_list_gp_q1",
                    "verified_for_single_task_gp_q1",
                }
            )
            assert isinstance(result["optimizer_requires_grad"], bool)
            assert isinstance(result["reasons"], list)
        if ACQUISITION_REGISTRY[acquisition_name].capabilities.one_shot:
            assert all(
                result["status"] == "requires_specialized_path" for result in optimizers.values()
            )


def test_classification_q1_evidence_is_scoped_to_verified_pairs() -> None:
    report = build_inventory()
    matrix = report["acquisition_optimizer_static"]
    acquisitions = {
        "BALD",
        "LatentStraddle",
        "MarginUncertainty",
        "PredictiveEntropy",
        "ProbabilityVariance",
    }
    for acquisition_name in acquisitions:
        for optimizer_name in ("de", "ga", "pso", "random", "sobol"):
            result = matrix[acquisition_name][optimizer_name]
            assert result["status"] == "verified_for_binary_single_task_gp_q1"
            assert result["runtime_verified"] is True
            assert len(result["evidence"]) == 1
            assert result["evidence"][0].startswith("tests/acquisition/")
    assert matrix["BALD"]["cmaes"]["runtime_verified"] is False
    assert matrix["qKnowledgeGradient"]["ga"]["runtime_verified"] is False


def test_botorch_q1_evidence_covers_five_classification_acquisitions() -> None:
    matrix = build_inventory()["acquisition_optimizer_static"]
    for acquisition_name in (
        "BALD",
        "LatentStraddle",
        "MarginUncertainty",
        "PredictiveEntropy",
        "ProbabilityVariance",
    ):
        verified = matrix[acquisition_name]["botorch"]
        assert verified["runtime_verified"] is True
        assert verified["evidence"] == [
            "tests/acquisition/test_classification_acqf_optimizer_smoke.py"
        ]
    assert matrix["qKnowledgeGradient"]["botorch"]["runtime_verified"] is False


def test_torch_and_hybrid_q1_evidence_is_scoped_to_predictive_entropy() -> None:
    matrix = build_inventory()["acquisition_optimizer_static"]
    for optimizer_name in ("hybrid", "torch_adam", "torch_adamw", "torch_sgd"):
        result = matrix["PredictiveEntropy"][optimizer_name]
        assert result["status"] == "verified_for_binary_single_task_gp_q1"
        assert result["runtime_verified"] is True
        assert len(result["evidence"]) == 1
        assert matrix["BALD"][optimizer_name]["runtime_verified"] is False


def test_runtime_evidence_coverage_counts_match_matrix() -> None:
    report = build_inventory()
    coverage = report["acquisition_optimizer_coverage"]
    matrix = report["acquisition_optimizer_static"]
    total = sum(len(results) for results in matrix.values())
    verified = sum(
        result["runtime_verified"] for results in matrix.values() for result in results.values()
    )
    assert coverage["total_pairs"] == total
    assert coverage["scoped_runtime_verified_pairs"] == verified
    assert coverage["unverified_or_specialized_pairs"] == total - verified
    assert len(coverage["verified_pairs"]) == verified
    assert report["verification"] == "static_inventory_with_scoped_runtime_evidence"


def test_unverified_triage_covers_exactly_remaining_pairs() -> None:
    report = build_inventory()
    matrix = report["acquisition_optimizer_static"]
    coverage = report["acquisition_optimizer_coverage"]
    unverified = coverage["unverified_pairs"]
    expected = {
        (acquisition_name, optimizer_name)
        for acquisition_name, results in matrix.items()
        for optimizer_name, result in results.items()
        if not result["runtime_verified"]
    }
    assert {(item["acquisition"], item["optimizer"]) for item in unverified} == expected
    assert len(unverified) == coverage["unverified_or_specialized_pairs"]
    assert {item["triage_group"] for item in unverified} <= {
        "classification_existing_model",
        "other_acquisitions",
        "specialized_path",
    }


def test_external_regression_evidence_is_separate_from_registered_matrix() -> None:
    report = build_inventory()
    evidence = report["external_botorch_acquisition_runtime_evidence"]
    assert set(evidence) == {"ExpectedImprovement", "UpperConfidenceBound"}
    for results in evidence.values():
        assert set(results) == {"botorch", "ga", "pso", "sobol"}
        assert all(result["runtime_verified"] for result in results.values())
        assert all(
            result["evidence"]
            == ["tests/optim/integration/test_optimizer_model_acquisition_e2e.py"]
            for result in results.values()
        )
    assert set(report["acquisition_optimizer_static"]) == set(report["acquisitions"])


def test_registered_regression_q1_runtime_evidence_is_scoped() -> None:
    report = build_inventory()
    matrix = report["acquisition_optimizer_static"]
    for acquisition_name in (
        "qLogExpectedImprovement",
        "qLogNoisyExpectedImprovement",
        "qProbabilityOfImprovement",
        "qSimpleRegret",
        "qUpperConfidenceBound",
    ):
        for optimizer_name in ("botorch", "ga", "pso", "sobol"):
            result = matrix[acquisition_name][optimizer_name]
            assert result["status"] == "verified_for_single_task_gp_q1"
            assert result["runtime_verified"]
            assert result["evidence"] == [
                "tests/optim/integration/test_optimizer_model_acquisition_e2e.py"
            ]
            assert "SingleTaskGP" in result["reasons"][0]


def test_multiobjective_q1_evidence_is_scoped() -> None:
    report = build_inventory()
    matrix = report["acquisition_optimizer_static"]
    for acquisition_name in (
        "qLogExpectedHypervolumeImprovement",
        "qLogNParEGO",
        "qLogNoisyExpectedHypervolumeImprovement",
    ):
        for optimizer_name in ("botorch", "ga", "pso", "sobol"):
            result = matrix[acquisition_name][optimizer_name]
            assert result["status"] == "verified_for_model_list_gp_q1"
            assert result["runtime_verified"]
            assert result["evidence"] == ["tests/acquisition/test_multiobjective.py"]
            assert "ModelListGP" in result["reasons"][0]

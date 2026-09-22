"""Contracts for BoTorch-native Bayesian optimization acquisition metadata."""

from robotorchan.acquisition.capabilities import AcquisitionPurpose
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.problem import ObjectiveType, OutputType, ProblemPurpose, ProblemSpec
from robotorchan.recommendation import recommend_compatible_workflows


def test_bo_acquisitions_are_metadata_only_botorch_entries() -> None:
    entry = ACQUISITION_REGISTRY["qLogExpectedImprovement"]

    assert entry.capabilities.purpose is AcquisitionPurpose.BAYESIAN_OPTIMIZATION
    assert entry.implementation_strategy == "BoTorch-native acquisition; no robotorchan wrapper"


def test_single_objective_bo_excludes_hypervolume_acquisitions() -> None:
    spec = ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION)

    names = {item.acquisition_name for item in recommend_compatible_workflows(spec)}

    assert "qLogExpectedImprovement" in names
    assert "qLogExpectedHypervolumeImprovement" not in names


def test_multi_objective_bo_selects_hypervolume_acquisitions() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        objective_type=ObjectiveType.MULTI,
        output_type=OutputType.MULTI,
    )

    names = {item.acquisition_name for item in recommend_compatible_workflows(spec)}

    assert "qLogExpectedHypervolumeImprovement" in names
    assert "qLogExpectedImprovement" not in names


def test_constrained_bo_filters_acquisition_metadata() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        constrained=True,
    )

    names = {item.acquisition_name for item in recommend_compatible_workflows(spec)}

    assert names == {
        "qLogExpectedImprovement",
        "qLogNoisyExpectedImprovement",
    }


def test_single_objective_multi_output_bo_keeps_scalarizable_mc_acquisitions() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        output_type=OutputType.MULTI,
    )

    names = {item.acquisition_name for item in recommend_compatible_workflows(spec)}

    assert "qLogExpectedImprovement" in names
    assert "qUpperConfidenceBound" in names

"""End-to-end contracts for the capability-aware workflow."""

from robotorchan.models.capabilities import InputType
from robotorchan.problem import OutputType, ProblemPurpose, ProblemSpec
from robotorchan.recommendation import recommend_compatible_workflows
from robotorchan.selector import evaluate_models


def test_mixed_high_dimensional_bo_flows_end_to_end() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        input_type=InputType.MIXED,
        high_dimensional=True,
    )

    evaluations = evaluate_models(spec)
    recommendations = recommend_compatible_workflows(spec)

    compatible = {item.model_name for item in evaluations if item.compatible}
    recommended = {item.model_name for item in recommendations}

    assert compatible
    assert recommended == compatible
    assert all(item.acquisition_name is None for item in recommendations)


def test_multi_output_active_learning_flows_end_to_end() -> None:
    spec = ProblemSpec(
        purpose=ProblemPurpose.ACTIVE_LEARNING,
        output_type=OutputType.MULTI,
    )

    recommendations = recommend_compatible_workflows(spec)

    assert recommendations
    assert all(item.acquisition_name is not None for item in recommendations)
    assert all(
        item.acquisition_name != "ExpectedPredictiveInformationGain" for item in recommendations
    )

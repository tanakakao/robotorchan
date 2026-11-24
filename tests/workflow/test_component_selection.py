"""Contracts for cross-component capability filtering."""

from robotorchan.acquisition import list_acquisitions
from robotorchan.models import list_models
from robotorchan.optim import list_optimizers, list_search_strategies
from robotorchan.workflow import ProblemSpec, SelectionStatus, evaluate_components


def test_every_factory_name_has_an_explicit_status() -> None:
    result = evaluate_components(ProblemSpec(purpose="bayesian_optimization"))
    assert tuple(item.name for item in result.models) == list_models()
    assert tuple(item.name for item in result.acquisitions) == list_acquisitions()
    assert tuple(item.name for item in result.optimizers) == list_optimizers()
    assert tuple(item.name for item in result.search_strategies) == list_search_strategies()


def test_mixed_batch_and_candidate_constraints() -> None:
    spec = ProblemSpec(
        purpose="active_learning", input_type="mixed", q=2, candidate_constrained=True
    )
    result = evaluate_components(spec)
    optimizers = {item.name: item for item in result.optimizers}
    assert optimizers["botorch"].status is SelectionStatus.CONDITIONAL
    assert "requires fixed_features_list" in optimizers["botorch"].reasons
    assert optimizers["torch_adam"].status is SelectionStatus.INCOMPATIBLE


def test_observation_kind_is_distinct_from_output_count() -> None:
    spec = ProblemSpec(purpose="active_learning", observation_kind="ordinal")
    result = evaluate_components(spec)
    models = {item.name: item for item in result.models}
    assert models["ordinal.standard"].status is SelectionStatus.COMPATIBLE
    assert models["multiclass.standard"].status is SelectionStatus.INCOMPATIBLE
    assert models["SingleTaskGP"].status is SelectionStatus.INCOMPATIBLE
    acquisitions = {item.name: item for item in result.acquisitions}
    assert acquisitions["qPredictiveEntropy"].status is SelectionStatus.CONDITIONAL


def test_unregistered_metadata_is_not_inferred_from_names() -> None:
    result = evaluate_components(ProblemSpec(purpose="bayesian_optimization"))
    acquisitions = {item.name: item for item in result.acquisitions}
    assert acquisitions["ExpectedImprovement"].status is SelectionStatus.UNKNOWN

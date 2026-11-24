"""Contracts for cross-component capability filtering."""

from robotorchan.acquisition import list_acquisitions
from robotorchan.models import list_models
from robotorchan.optim import list_optimizers, list_search_strategies
from robotorchan.workflow import (
    ProblemSpec,
    SelectionStatus,
    evaluate_components,
    select_compatible_models,
)
from robotorchan.workflow.components import evaluate_component_inventory


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


def test_lookahead_metadata_is_registered() -> None:
    result = evaluate_components(ProblemSpec(purpose="bayesian_optimization"))
    acquisitions = {item.name: item for item in result.acquisitions}
    assert acquisitions["qMultiStepLookahead"].status is SelectionStatus.CONDITIONAL


def test_all_search_strategies_have_explicit_capability_checks() -> None:
    result = evaluate_components(ProblemSpec(purpose="bayesian_optimization"))
    assert all(item.status is not SelectionStatus.UNKNOWN for item in result.search_strategies)


def test_specialist_and_strategy_constraints() -> None:
    spec = ProblemSpec(
        purpose="bayesian_optimization", input_type="mixed", q=2, candidate_constrained=True
    )
    result = evaluate_components(spec)
    strategies = {item.name: item for item in result.search_strategies}
    optimizers = {item.name: item for item in result.optimizers}
    assert strategies["DiscreteTrustRegionStrategy"].status is SelectionStatus.INCOMPATIBLE
    assert strategies["REMBOStrategy"].status is SelectionStatus.INCOMPATIBLE
    assert strategies["TuRBOStrategy"].status is SelectionStatus.CONDITIONAL
    assert optimizers["optimize_acqf_mixed_ga"].status is SelectionStatus.CONDITIONAL
    assert optimizers["optimize_acqf_ga"].status is SelectionStatus.INCOMPATIBLE
    assert optimizers["optimize_vector_nsga2_mixed"].status is SelectionStatus.INCOMPATIBLE


def test_native_analytic_and_mc_acquisition_batch_limits() -> None:
    result = evaluate_components(ProblemSpec(purpose="bayesian_optimization", q=2))
    acquisitions = {item.name: item for item in result.acquisitions}
    assert acquisitions["ExpectedImprovement"].status is SelectionStatus.INCOMPATIBLE
    assert acquisitions["qExpectedImprovement"].status is SelectionStatus.CONDITIONAL


def test_extended_classification_model_selection() -> None:
    spec = ProblemSpec(purpose="active_learning", observation_kind="ordinal", high_dimensional=True)
    result = evaluate_components(spec)
    models = {item.name: item for item in result.models}
    assert models["PCAOrdinalGPClassifier"].status is SelectionStatus.COMPATIBLE
    assert models["PCAMulticlassGPClassifier"].status is SelectionStatus.INCOMPATIBLE
    assert models["RandomForestOrdinalClassifier"].status is SelectionStatus.INCOMPATIBLE


def test_remaining_specialist_optimizer_contracts() -> None:
    result = evaluate_components(ProblemSpec(purpose="bayesian_optimization", q=2))
    optimizers = {item.name: item for item in result.optimizers}
    assert all(item.status is not SelectionStatus.UNKNOWN for item in result.optimizers)
    assert optimizers["optimize_mixed_one_shot_acqf"].status is SelectionStatus.INCOMPATIBLE
    assert optimizers["optimize_safe_acqf"].status is SelectionStatus.CONDITIONAL
    assert optimizers["ContextualOptimizer"].status is SelectionStatus.CONDITIONAL


def test_specialized_acquisition_requirements() -> None:
    result = evaluate_components(ProblemSpec(purpose="bayesian_optimization", q=2))
    acquisitions = {item.name: item for item in result.acquisitions}
    assert acquisitions["qMaxValueEntropy"].status is SelectionStatus.INCOMPATIBLE
    assert acquisitions["qMultiFidelityMaxValueEntropy"].status is SelectionStatus.INCOMPATIBLE
    assert acquisitions["qExpectedUtilityOfBestOption"].status is SelectionStatus.INCOMPATIBLE
    assert acquisitions["qJointEntropySearch"].status is SelectionStatus.CONDITIONAL


def test_model_selector_agrees_with_component_selection() -> None:
    for kind in ("regression", "binary", "multiclass", "ordinal"):
        spec = ProblemSpec(purpose="active_learning", observation_kind=kind)
        selected = select_compatible_models(spec)
        statuses = {item.name: item.status for item in evaluate_components(spec).models}
        assert selected
        assert all(statuses[name] is SelectionStatus.COMPATIBLE for name in selected)
        if kind != "regression":
            assert all(name.startswith(f"{kind}.") for name in selected)


def test_unknown_inventory_entries_remain_explicit() -> None:
    result = evaluate_component_inventory(
        ProblemSpec(purpose="bayesian_optimization"),
        models=("unregistered_model",),
        acquisitions=("unregistered_acquisition",),
        optimizers=("unregistered_optimizer",),
        search_strategies=("unregistered_strategy",),
    )
    for group in (result.models, result.acquisitions, result.optimizers, result.search_strategies):
        assert group[0].status is SelectionStatus.UNKNOWN
        assert group[0].reasons


def test_special_model_contracts() -> None:
    spec = ProblemSpec(purpose="bayesian_optimization", input_type="mixed")
    models = {item.name: item for item in evaluate_components(spec).models}
    for name in ("BOCSSurrogate", "COMBOSurrogate", "RGPEModel", "InterventionGPSurrogate"):
        assert models[name].status is SelectionStatus.CONDITIONAL
    assert models["GPStructuralCausalModel"].status is SelectionStatus.CONDITIONAL
    assert models["CalibratedOrdinalClassifier"].status is SelectionStatus.INCOMPATIBLE
    spec = ProblemSpec(purpose="active_learning", observation_kind="ordinal")
    models = {item.name: item for item in evaluate_components(spec).models}
    assert models["CalibratedOrdinalClassifier"].status is SelectionStatus.CONDITIONAL
    assert models["BOCSSurrogate"].status is SelectionStatus.INCOMPATIBLE

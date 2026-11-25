"""Contracts for cross-component capability filtering."""

from robotorchan.acquisition import list_acquisitions
from robotorchan.models import list_models
from robotorchan.optim import list_optimizers, list_search_strategies
from robotorchan.workflow import (
    ProblemSpec,
    SelectionStatus,
    check_compatibility,
    evaluate_components,
    recommend,
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


def test_classifier_feasibility_acquisition_conditions() -> None:
    spec = ProblemSpec(purpose="bayesian_optimization", observation_kind="binary")
    acquisitions = {item.name: item for item in evaluate_components(spec).acquisitions}
    assert (
        acquisitions["ClassificationProbabilityAcquisition"].status is SelectionStatus.CONDITIONAL
    )
    assert acquisitions["FeasibilityWeightedAcquisition"].status is SelectionStatus.CONDITIONAL
    spec = ProblemSpec(purpose="bayesian_optimization", q=2, constrained=True)
    acquisitions = {item.name: item for item in evaluate_components(spec).acquisitions}
    assert (
        acquisitions["ClassificationProbabilityAcquisition"].status is SelectionStatus.INCOMPATIBLE
    )
    assert acquisitions["FeasibilityWeightedAcquisition"].status is SelectionStatus.CONDITIONAL
    assert "does not support pending candidates or joint fantasization" in (
        acquisitions["FeasibilityWeightedAcquisition"].reasons
    )


def test_additional_native_acquisition_contracts() -> None:
    result = evaluate_components(ProblemSpec(purpose="bayesian_optimization", q=2))
    acquisitions = {item.name: item.status for item in result.acquisitions}
    for name in (
        "ConstrainedExpectedImprovement",
        "LogConstrainedExpectedImprovement",
        "PosteriorStandardDeviation",
    ):
        if name in acquisitions:
            assert acquisitions[name] is SelectionStatus.INCOMPATIBLE
    result = evaluate_components(ProblemSpec(purpose="active_learning", q=2))
    acquisitions = {item.name: item.status for item in result.acquisitions}
    if "qNegIntegratedPosteriorVariance" in acquisitions:
        assert acquisitions["qNegIntegratedPosteriorVariance"] is SelectionStatus.CONDITIONAL


def test_report_remaining_unknown_components(capsys) -> None:
    result = evaluate_components(ProblemSpec(purpose="bayesian_optimization"))
    for category in ("models", "acquisitions", "optimizers", "search_strategies"):
        entries = getattr(result, category)
        unknown = [item.name for item in entries if item.status is SelectionStatus.UNKNOWN]
        with capsys.disabled():
            print(f"{category}: {len(unknown)}/{len(entries)} unknown: {unknown}")
        assert not unknown, f"{category}: missing capability checks: {unknown}"


def test_generic_binary_ensemble_and_causal_boundaries() -> None:
    spec = ProblemSpec(purpose="active_learning", observation_kind="binary")
    models = {item.name: item for item in evaluate_components(spec).models}
    for name in ("GPBinaryClassificationEnsemble", "HeterogeneousBinaryClassificationEnsemble"):
        assert models[name].status in (SelectionStatus.COMPATIBLE, SelectionStatus.CONDITIONAL)
    assert models["GPStructuralCausalModel"].status is SelectionStatus.INCOMPATIBLE
    assert models["StructuralCausalModel"].status is SelectionStatus.INCOMPATIBLE


def test_remaining_native_acquisition_conditions() -> None:
    result = evaluate_components(ProblemSpec(purpose="bayesian_optimization", q=2))
    acquisitions = {item.name: item.status for item in result.acquisitions}
    assert acquisitions["LogProbabilityOfImprovement"] is SelectionStatus.INCOMPATIBLE
    assert acquisitions["qMultiFidelityLowerBoundMaxValueEntropy"] is SelectionStatus.INCOMPATIBLE
    assert acquisitions["ProjectedAcquisitionFunction"] is SelectionStatus.CONDITIONAL
    result = evaluate_components(ProblemSpec(purpose="active_learning", q=2, preference=True))
    acquisitions = {item.name: item.status for item in result.acquisitions}
    assert acquisitions["PairwiseMCPosteriorVariance"] is SelectionStatus.CONDITIONAL


def test_workflow_compatibility_and_explicit_recommendation() -> None:
    spec = ProblemSpec(purpose="bayesian_optimization", q=2)
    result = check_compatibility(
        spec, model="SingleTaskGP", acquisition="qLogExpectedImprovement", optimizer="botorch"
    )
    assert result.status is SelectionStatus.CONDITIONAL
    assert result.conditions
    recommendations = recommend(spec, model="SingleTaskGP", acquisition="qLogExpectedImprovement")
    assert len(recommendations) == 1
    assert recommendations[0].model_name == "SingleTaskGP"
    assert not recommend(spec, model="SingleTaskGP", acquisition="ExpectedImprovement")


def test_workflow_unknown_and_one_shot_rejection() -> None:
    spec = ProblemSpec(purpose="bayesian_optimization")
    result = check_compatibility(spec, model="unknown", acquisition="qLogExpectedImprovement")
    assert result.status is SelectionStatus.UNKNOWN
    result = check_compatibility(
        spec, model="SingleTaskGP", acquisition="qKnowledgeGradient", optimizer="sobol"
    )
    assert result.status is SelectionStatus.INCOMPATIBLE


def test_workflow_classification_class_name() -> None:
    spec = ProblemSpec(purpose="active_learning", observation_kind="multiclass")
    result = check_compatibility(
        spec, model="MulticlassSingleTaskGPClassifier", acquisition="PredictiveEntropy"
    )
    assert result.status is SelectionStatus.CONDITIONAL


def test_recommendation_preserves_class_name_and_special_acquisition() -> None:
    spec = ProblemSpec(purpose="active_learning", observation_kind="multiclass")
    results = recommend(spec, model="MulticlassSingleTaskGPClassifier")
    assert results
    assert all(item.model_name == "MulticlassSingleTaskGPClassifier" for item in results)
    spec = ProblemSpec(purpose="bayesian_optimization")
    results = recommend(spec, acquisition="LogProbabilityOfImprovement")
    assert results
    assert all(item.acquisition_name == "LogProbabilityOfImprovement" for item in results)


def test_strategy_owns_optimizer_selection() -> None:
    spec = ProblemSpec(purpose="bayesian_optimization")
    kwargs = dict(model="SingleTaskGP", acquisition="qLogExpectedImprovement")
    assert not recommend(spec, **kwargs, optimizer="ga", search_strategy="OriginalSpaceStrategy")
    results = recommend(spec, **kwargs, search_strategy="SobolSearchStrategy")
    assert results
    assert results[0].optimizer_name is None
    results = recommend(spec, **kwargs, optimizer="ga", search_strategy="TuRBOStrategy")
    assert results
    assert results[0].optimizer_name == "ga"


def test_auto_optimizer_and_strategy_candidates() -> None:
    spec = ProblemSpec(purpose="bayesian_optimization")
    kwargs = dict(model="SingleTaskGP", acquisition="qLogExpectedImprovement")
    results = recommend(spec, **kwargs, optimizer="auto")
    names = {item.optimizer_name for item in results}
    assert {"botorch", "ga", "sobol"} <= names
    assert "optimize_vector_nsga2" not in names
    assert "optimize_acqf_mixed_botorch" not in names
    results = recommend(spec, **kwargs, search_strategy="auto")
    names = {item.search_strategy_name for item in results}
    assert {"OriginalSpaceStrategy", "SobolSearchStrategy", "TuRBOStrategy"} <= names
    assert "MixedSpaceStrategy" not in names
    assert all(item.optimizer_name is None for item in results)


def test_causal_models_require_dedicated_workflow() -> None:
    spec = ProblemSpec(purpose="bayesian_optimization")
    for model in ("GPStructuralCausalModel", "StructuralCausalModel", "InterventionGPSurrogate"):
        result = check_compatibility(spec, model=model, acquisition="qLogExpectedImprovement")
        assert result.status is SelectionStatus.INCOMPATIBLE


def test_auto_model_and_acquisition_factory_inventory() -> None:
    spec = ProblemSpec(purpose="bayesian_optimization")
    results = recommend(spec, model="auto", acquisition="qLogExpectedImprovement")
    names = {item.model_name for item in results}
    assert "SingleTaskGP" in names
    assert "RGPEModel" in names
    assert "StructuralCausalModel" not in names
    results = recommend(spec, model="SingleTaskGP", acquisition="auto")
    names = {item.acquisition_name for item in results}
    assert "qLogExpectedImprovement" in names
    assert "LogProbabilityOfImprovement" in names


def test_recommendation_required_arguments() -> None:
    spec = ProblemSpec(purpose="bayesian_optimization")
    result = recommend(spec, model="SingleTaskGP", acquisition="LogExpectedImprovement")[0]
    arguments = result.required_arguments()
    assert {"train_X", "train_Y"} <= set(arguments["model"])
    assert {"model", "best_f"} <= set(arguments["acquisition"])
    assert {"acq_function", "bounds", "q"} <= set(arguments["optimizer"])

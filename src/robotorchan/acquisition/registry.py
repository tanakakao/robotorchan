"""Capability registry for robotorchan acquisition extensions."""

from robotorchan.acquisition.capabilities import (
    AcquisitionCapabilities,
    AcquisitionPurpose,
    AcquisitionRegistryEntry,
    PosteriorRequirement,
)


def _active_learning(
    name: str,
    *,
    posterior_requirement: PosteriorRequirement = PosteriorRequirement.MARGINAL_MOMENTS,
    supports_multi_output: bool = True,
    requires_single_output: bool = False,
    limitations: tuple[str, ...] = (),
) -> AcquisitionRegistryEntry:
    return AcquisitionRegistryEntry(
        acquisition_name=name,
        capabilities=AcquisitionCapabilities(
            purpose=AcquisitionPurpose.ACTIVE_LEARNING,
            posterior_requirement=posterior_requirement,
            max_q=1,
            supports_multi_output=supports_multi_output,
            requires_single_output=requires_single_output,
        ),
        implementation_strategy="robotorchan active-learning acquisition",
        limitations=limitations,
    )


ACQUISITION_REGISTRY: dict[str, AcquisitionRegistryEntry] = {
    name: _active_learning(
        name,
        limitations=(
            "ensemble posteriors are not supported",
            "structured outputs require scalarization",
        ),
    )
    for name in (
        "BoundaryVariance",
        "PosteriorStd",
        "PosteriorVariance",
        "RandomizedStraddle",
        "Straddle",
    )
}
ACQUISITION_REGISTRY["ExpectedPredictiveInformationGain"] = _active_learning(
    "ExpectedPredictiveInformationGain",
    posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
    supports_multi_output=False,
    requires_single_output=True,
    limitations=(
        "requires a single-output Gaussian posterior",
        "ensemble posteriors are not supported",
    ),
)


def get_acquisition_registry_entry(acquisition_name: str) -> AcquisitionRegistryEntry:
    """Return capability metadata for a registered acquisition extension."""
    return ACQUISITION_REGISTRY[acquisition_name]


def _botorch_bo(
    name: str,
    *,
    supports_constraints: bool = False,
    supports_multi_objective: bool = False,
    supports_multi_output: bool = True,
    requires_fantasize: bool = False,
    requires_multi_fidelity: bool = False,
    one_shot: bool = False,
) -> AcquisitionRegistryEntry:
    return AcquisitionRegistryEntry(
        acquisition_name=name,
        capabilities=AcquisitionCapabilities(
            purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
            posterior_requirement=PosteriorRequirement.POSTERIOR_SAMPLES,
            max_q=None,
            supports_multi_output=supports_multi_output,
            supports_ensemble=True,
            supports_constraints=supports_constraints,
            supports_multi_objective=supports_multi_objective,
            monte_carlo=True,
            requires_fantasize=requires_fantasize,
            requires_multi_fidelity=requires_multi_fidelity,
            one_shot=one_shot,
        ),
        implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    )


ACQUISITION_REGISTRY.update(
    {
        "qLogExpectedImprovement": _botorch_bo(
            "qLogExpectedImprovement",
            supports_constraints=True,
        ),
        "qLogNoisyExpectedImprovement": _botorch_bo(
            "qLogNoisyExpectedImprovement",
            supports_constraints=True,
        ),
        "qUpperConfidenceBound": _botorch_bo("qUpperConfidenceBound"),
        "qKnowledgeGradient": _botorch_bo(
            "qKnowledgeGradient",
            supports_multi_output=False,
            requires_fantasize=True,
            one_shot=True,
        ),
        "qMultiFidelityKnowledgeGradient": _botorch_bo(
            "qMultiFidelityKnowledgeGradient",
            supports_multi_output=False,
            requires_fantasize=True,
            requires_multi_fidelity=True,
            one_shot=True,
        ),
        "qLogExpectedHypervolumeImprovement": _botorch_bo(
            "qLogExpectedHypervolumeImprovement",
            supports_constraints=True,
            supports_multi_objective=True,
        ),
        "qLogNoisyExpectedHypervolumeImprovement": _botorch_bo(
            "qLogNoisyExpectedHypervolumeImprovement",
            supports_constraints=True,
            supports_multi_objective=True,
        ),
        "qLogNParEGO": _botorch_bo(
            "qLogNParEGO",
            supports_constraints=True,
            supports_multi_objective=True,
        ),
    }
)

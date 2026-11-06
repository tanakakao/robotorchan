"""Capability registry for robotorchan acquisition extensions."""

from robotorchan.acquisition.capabilities import (
    AcquisitionCapabilities,
    AcquisitionPurpose,
    AcquisitionRegistryEntry,
    AcquisitionTarget,
    PosteriorRequirement,
)
from robotorchan.models.capabilities import ObservationType


def _active_learning(
    name: str,
    *,
    posterior_requirement: PosteriorRequirement = PosteriorRequirement.MARGINAL_MOMENTS,
    supports_multi_output: bool = True,
    requires_single_output: bool = False,
    limitations: tuple[str, ...] = (),
    max_q: int | None = 1,
) -> AcquisitionRegistryEntry:
    return AcquisitionRegistryEntry(
        acquisition_name=name,
        capabilities=AcquisitionCapabilities(
            purpose=AcquisitionPurpose.ACTIVE_LEARNING,
            posterior_requirement=posterior_requirement,
            max_q=max_q,
            supports_multi_output=supports_multi_output,
            requires_single_output=requires_single_output,
        ),
        implementation_strategy="robotorchan active-learning acquisition",
        limitations=limitations,
    )


ACQUISITION_REGISTRY: dict[str, AcquisitionRegistryEntry] = {}
for name in (
    "qBoundaryVariance",
    "qRandomizedStraddle",
    "qStraddle",
    "qPosteriorStd",
    "qPosteriorVariance",
):
    ACQUISITION_REGISTRY[name] = _active_learning(
        name,
        max_q=None,
        limitations=(
            "q>1 aggregates marginal scores without candidate diversity",
            "ensemble posteriors are not supported",
            "structured outputs require scalarization",
        ),
    )

ACQUISITION_REGISTRY["qJointPosteriorEntropy"] = AcquisitionRegistryEntry(
    acquisition_name="qJointPosteriorEntropy",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.ACTIVE_LEARNING,
        posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
        max_q=None,
        supports_multi_output=False,
        requires_single_output=True,
    ),
    implementation_strategy="robotorchan joint Gaussian batch entropy",
    limitations=("requires a single-output joint Gaussian posterior",),
)
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


def _classification_active_learning(
    name: str,
    *,
    target: AcquisitionTarget,
    posterior_requirement: PosteriorRequirement,
    supports_ensemble: bool = False,
    max_q: int | None = 1,
) -> AcquisitionRegistryEntry:
    return AcquisitionRegistryEntry(
        acquisition_name=name,
        capabilities=AcquisitionCapabilities(
            purpose=AcquisitionPurpose.ACTIVE_LEARNING,
            posterior_requirement=posterior_requirement,
            observation_types=frozenset({ObservationType.CLASSIFICATION}),
            target=target,
            max_q=max_q,
            supports_multi_output=False,
            supports_ensemble=supports_ensemble,
        ),
        implementation_strategy="robotorchan classification active-learning acquisition",
    )


ACQUISITION_REGISTRY.update(
    {
        "PredictiveEntropy": _classification_active_learning(
            "PredictiveEntropy",
            target=AcquisitionTarget.LABEL_UNCERTAINTY,
            posterior_requirement=PosteriorRequirement.MARGINAL_MOMENTS,
            supports_ensemble=True,
        ),
        "MarginUncertainty": _classification_active_learning(
            "MarginUncertainty",
            target=AcquisitionTarget.CLASS_PROBABILITY,
            posterior_requirement=PosteriorRequirement.MARGINAL_MOMENTS,
            supports_ensemble=True,
        ),
        "ProbabilityVariance": _classification_active_learning(
            "ProbabilityVariance",
            target=AcquisitionTarget.CLASS_PROBABILITY,
            posterior_requirement=PosteriorRequirement.POSTERIOR_SAMPLES,
            supports_ensemble=True,
        ),
        "BALD": _classification_active_learning(
            "BALD",
            target=AcquisitionTarget.LABEL_UNCERTAINTY,
            posterior_requirement=PosteriorRequirement.POSTERIOR_SAMPLES,
            supports_ensemble=True,
        ),
        "qMarginUncertainty": _classification_active_learning(
            "qMarginUncertainty",
            target=AcquisitionTarget.CLASS_PROBABILITY,
            posterior_requirement=PosteriorRequirement.MARGINAL_MOMENTS,
            supports_ensemble=True,
            max_q=None,
        ),
        "qPredictiveEntropy": _classification_active_learning(
            "qPredictiveEntropy",
            target=AcquisitionTarget.LABEL_UNCERTAINTY,
            posterior_requirement=PosteriorRequirement.MARGINAL_MOMENTS,
            supports_ensemble=True,
            max_q=None,
        ),
        "LatentStraddle": _classification_active_learning(
            "LatentStraddle",
            target=AcquisitionTarget.LATENT,
            posterior_requirement=PosteriorRequirement.MARGINAL_MOMENTS,
        ),
    }
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
        "qProbabilityOfImprovement": _botorch_bo(
            "qProbabilityOfImprovement",
            supports_constraints=True,
        ),
        "qSimpleRegret": _botorch_bo("qSimpleRegret"),
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

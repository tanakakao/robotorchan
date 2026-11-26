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
ACQUISITION_REGISTRY["qConditionalExpectedPredictiveInformationGain"] = _active_learning(
    "qConditionalExpectedPredictiveInformationGain",
    max_q=None,
    posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
    supports_multi_output=False,
    requires_single_output=True,
    limitations=(
        "pending candidate locations are optional; a joint Gaussian posterior is required",
        "does not model uncertain pending outcomes or fantasization",
    ),
)
ACQUISITION_REGISTRY["qJointExpectedPredictiveInformationGain"] = _active_learning(
    "qJointExpectedPredictiveInformationGain",
    max_q=None,
    posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
    supports_multi_output=False,
    requires_single_output=True,
    limitations=(
        "requires a single-output Gaussian posterior",
        "ensemble posteriors are not supported",
    ),
)
ACQUISITION_REGISTRY["qExpectedPredictiveInformationGain"] = _active_learning(
    "qExpectedPredictiveInformationGain",
    max_q=None,
    posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
    supports_multi_output=False,
    requires_single_output=True,
    limitations=(
        "mean marginal information gain, not joint batch information",
        "requires a single-output Gaussian posterior",
        "ensemble posteriors are not supported",
    ),
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
        "qBALD": _classification_active_learning(
            "qBALD",
            target=AcquisitionTarget.LABEL_UNCERTAINTY,
            posterior_requirement=PosteriorRequirement.POSTERIOR_SAMPLES,
            max_q=8,
        ),
        "qMarginUncertainty": _classification_active_learning(
            "qMarginUncertainty",
            target=AcquisitionTarget.CLASS_PROBABILITY,
            posterior_requirement=PosteriorRequirement.MARGINAL_MOMENTS,
            supports_ensemble=True,
            max_q=None,
        ),
        "qProbabilityVariance": _classification_active_learning(
            "qProbabilityVariance",
            target=AcquisitionTarget.CLASS_PROBABILITY,
            posterior_requirement=PosteriorRequirement.POSTERIOR_SAMPLES,
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
        "qLatentStraddle": _classification_active_learning(
            "qLatentStraddle",
            target=AcquisitionTarget.LATENT,
            posterior_requirement=PosteriorRequirement.MARGINAL_MOMENTS,
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


# Native analytic acquisitions have a single-candidate contract.
for name in (
    "ExpectedImprovement",
    "LogExpectedImprovement",
    "ProbabilityOfImprovement",
    "UpperConfidenceBound",
    "PosteriorMean",
    "NoisyExpectedImprovement",
    "LogNoisyExpectedImprovement",
):
    ACQUISITION_REGISTRY[name] = AcquisitionRegistryEntry(
        acquisition_name=name,
        capabilities=AcquisitionCapabilities(
            purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
            posterior_requirement=PosteriorRequirement.MARGINAL_MOMENTS,
            max_q=1,
            requires_single_output=True,
        ),
        implementation_strategy="BoTorch-native analytic acquisition",
        limitations=("requires a compatible scalar posterior or posterior transform",),
    )

ACQUISITION_REGISTRY.update(
    {
        "qExpectedImprovement": _botorch_bo("qExpectedImprovement", supports_constraints=True),
        "qNoisyExpectedImprovement": _botorch_bo(
            "qNoisyExpectedImprovement", supports_constraints=True
        ),
        "qExpectedHypervolumeImprovement": _botorch_bo(
            "qExpectedHypervolumeImprovement",
            supports_constraints=True,
            supports_multi_objective=True,
        ),
        "qNoisyExpectedHypervolumeImprovement": _botorch_bo(
            "qNoisyExpectedHypervolumeImprovement",
            supports_constraints=True,
            supports_multi_objective=True,
        ),
    }
)


# Specialized native acquisitions retain their original constructor contracts.
ACQUISITION_REGISTRY["qMultiStepLookahead"] = AcquisitionRegistryEntry(
    acquisition_name="qMultiStepLookahead",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.POSTERIOR_SAMPLES,
        requires_single_output=True,
        requires_fantasize=True,
        one_shot=True,
    ),
    implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    limitations=("requires method-specific constructor inputs and posterior checks",),
)

ACQUISITION_REGISTRY["qMaxValueEntropy"] = AcquisitionRegistryEntry(
    acquisition_name="qMaxValueEntropy",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
        max_q=1,
        requires_single_output=True,
        requires_fantasize=True,
    ),
    implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    limitations=("requires method-specific constructor inputs and posterior checks",),
)

ACQUISITION_REGISTRY["qMultiFidelityMaxValueEntropy"] = AcquisitionRegistryEntry(
    acquisition_name="qMultiFidelityMaxValueEntropy",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
        max_q=1,
        requires_single_output=True,
        requires_fantasize=True,
        requires_multi_fidelity=True,
    ),
    implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    limitations=("requires method-specific constructor inputs and posterior checks",),
)

ACQUISITION_REGISTRY["qJointEntropySearch"] = AcquisitionRegistryEntry(
    acquisition_name="qJointEntropySearch",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
        requires_single_output=True,
    ),
    implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    limitations=("requires method-specific constructor inputs and posterior checks",),
)

ACQUISITION_REGISTRY["qPredictiveEntropySearch"] = AcquisitionRegistryEntry(
    acquisition_name="qPredictiveEntropySearch",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
        requires_single_output=True,
    ),
    implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    limitations=("requires method-specific constructor inputs and posterior checks",),
)

ACQUISITION_REGISTRY["qLowerBoundMultiObjectiveJointEntropySearch"] = AcquisitionRegistryEntry(
    acquisition_name="qLowerBoundMultiObjectiveJointEntropySearch",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
        supports_multi_output=True,
        supports_multi_objective=True,
    ),
    implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    limitations=("requires method-specific constructor inputs and posterior checks",),
)

ACQUISITION_REGISTRY["qMultiObjectivePredictiveEntropySearch"] = AcquisitionRegistryEntry(
    acquisition_name="qMultiObjectivePredictiveEntropySearch",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
        supports_multi_output=True,
        supports_multi_objective=True,
    ),
    implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    limitations=("requires method-specific constructor inputs and posterior checks",),
)

ACQUISITION_REGISTRY["qLowerBoundMultiObjectiveMaxValueEntropySearch"] = AcquisitionRegistryEntry(
    acquisition_name="qLowerBoundMultiObjectiveMaxValueEntropySearch",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
        supports_multi_output=True,
        supports_multi_objective=True,
    ),
    implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    limitations=("requires method-specific constructor inputs and posterior checks",),
)

ACQUISITION_REGISTRY["AnalyticExpectedUtilityOfBestOption"] = AcquisitionRegistryEntry(
    acquisition_name="AnalyticExpectedUtilityOfBestOption",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
        max_q=2,
        requires_single_output=True,
        requires_preference=True,
    ),
    implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    limitations=("requires method-specific constructor inputs and posterior checks",),
)

ACQUISITION_REGISTRY["qExpectedUtilityOfBestOption"] = AcquisitionRegistryEntry(
    acquisition_name="qExpectedUtilityOfBestOption",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.POSTERIOR_SAMPLES,
        requires_single_output=True,
        requires_preference=True,
    ),
    implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    limitations=("requires method-specific constructor inputs and posterior checks",),
)

ACQUISITION_REGISTRY["PairwiseBayesianActiveLearningByDisagreement"] = AcquisitionRegistryEntry(
    acquisition_name="PairwiseBayesianActiveLearningByDisagreement",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.ACTIVE_LEARNING,
        posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
        max_q=2,
        requires_single_output=True,
        requires_preference=True,
    ),
    implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    limitations=("requires method-specific constructor inputs and posterior checks",),
)

ACQUISITION_REGISTRY["ExpectedHypervolumeImprovement"] = AcquisitionRegistryEntry(
    acquisition_name="ExpectedHypervolumeImprovement",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.MARGINAL_MOMENTS,
        max_q=1,
        supports_multi_output=True,
        supports_multi_objective=True,
    ),
    implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
    limitations=("requires method-specific constructor inputs and posterior checks",),
)


# Native hypervolume KG requires independent output-wise fantasy conditioning.
for name in ("qHypervolumeKnowledgeGradient", "qMultiFidelityHypervolumeKnowledgeGradient"):
    ACQUISITION_REGISTRY[name] = AcquisitionRegistryEntry(
        acquisition_name=name,
        capabilities=AcquisitionCapabilities(
            purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
            posterior_requirement=PosteriorRequirement.POSTERIOR_SAMPLES,
            supports_multi_output=True,
            supports_multi_objective=True,
            monte_carlo=True,
            requires_fantasize=True,
            requires_multi_fidelity=name == "qMultiFidelityHypervolumeKnowledgeGradient",
            one_shot=True,
        ),
        implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper",
        limitations=(
            "requires ModelList with masked fantasy support and a reference point",
            "requires one-shot optimization with augmented candidates",
            "joint masked optimization supported; pending workflow remains unverified",
        ),
    )


ACQUISITION_REGISTRY["qDecoupledKnowledgeGradient"] = AcquisitionRegistryEntry(
    acquisition_name="qDecoupledKnowledgeGradient",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.POSTERIOR_SAMPLES,
        supports_multi_output=True,
        monte_carlo=True,
        requires_fantasize=True,
    ),
    implementation_strategy="robotorchan finite-decision-set masked fantasy acquisition",
    limitations=(
        "requires independent ModelListGP outputs and a fixed terminal decision set",
        "requires one fantasy sampler per output and a boolean evaluation mask",
        "optional target fidelity projection; independent Gaussian constraint utility",
        "pending evaluations are not supported",
    ),
)


ACQUISITION_REGISTRY["qDecoupledExpectedPredictiveInformationGain"] = _active_learning(
    "qDecoupledExpectedPredictiveInformationGain",
    posterior_requirement=PosteriorRequirement.JOINT_GAUSSIAN,
    max_q=None,
    limitations=(
        "requires independent ModelListGP outputs and positive observation noise",
        "requires output weights and an evaluation mask for candidate rows",
        "optional target fidelity projection and point-dependent observation variance",
        "pending evaluations and correlated outputs are not supported",
    ),
)


ACQUISITION_REGISTRY["qDecoupledActiveLearning"] = AcquisitionRegistryEntry(
    acquisition_name="qDecoupledActiveLearning",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.ACTIVE_LEARNING,
        posterior_requirement=PosteriorRequirement.MARGINAL_MOMENTS,
        observation_types=frozenset({ObservationType.REGRESSION, ObservationType.CLASSIFICATION}),
        supports_multi_output=True,
        requires_heterogeneous_model=True,
    ),
    implementation_strategy="robotorchan masked heterogeneous acquisition composition",
    limitations=(
        "requires HeterogeneousModel with one semantic output per entry",
        "posterior requirements and q limits depend on the supplied child acquisitions",
        "requires explicit utility weights and evaluation masks",
        "pending evaluations are not supported",
    ),
)


ACQUISITION_REGISTRY["qDecoupledParticleKnowledgeGradient"] = AcquisitionRegistryEntry(
    acquisition_name="qDecoupledParticleKnowledgeGradient",
    capabilities=AcquisitionCapabilities(
        purpose=AcquisitionPurpose.BAYESIAN_OPTIMIZATION,
        posterior_requirement=PosteriorRequirement.POSTERIOR_SAMPLES,
        observation_types=frozenset({ObservationType.REGRESSION, ObservationType.CLASSIFICATION}),
        supports_multi_output=True,
        requires_heterogeneous_model=True,
        requires_derivative_free=True,
        supports_constraints=True,
    ),
    implementation_strategy="robotorchan likelihood-weighted particle KG",
    limitations=(
        "finite decision set and independent semantic outputs",
        "requires explicit sample utility and Gaussian observation variances",
        "approximate posterior update; derivative-free optimization only",
        "pending evaluations are not supported",
    ),
)

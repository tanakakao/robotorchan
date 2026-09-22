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

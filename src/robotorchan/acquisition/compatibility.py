"""Compatibility checks between registered models and acquisitions."""

from dataclasses import dataclass
from enum import StrEnum

from robotorchan.acquisition.capabilities import PosteriorRequirement
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models.registry import MODEL_REGISTRY


class CompatibilityStatus(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    status: CompatibilityStatus
    reasons: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        """Whether the model/acquisition pair is compatible."""
        return self.status is CompatibilityStatus.COMPATIBLE


def check_model_acquisition_compatibility(
    model_name: str,
    acquisition_name: str,
) -> CompatibilityResult:
    """Check static compatibility using registry metadata only."""
    model = MODEL_REGISTRY[model_name]
    acquisition = ACQUISITION_REGISTRY[acquisition_name]
    model_capabilities = model.capabilities
    acquisition_capabilities = acquisition.capabilities
    reasons: list[str] = []

    if model_capabilities.non_gp and not acquisition_capabilities.monte_carlo:
        reason = "non-GP empirical ensembles require BoTorch Monte Carlo acquisitions"
        reasons.append(reason)
    if acquisition_capabilities.requires_fantasize and not model_capabilities.supports_fantasize:
        reasons.append("acquisition requires fantasy-model support")
    if model_capabilities.ensemble_posterior and not acquisition_capabilities.supports_ensemble:
        reasons.append("acquisition does not support ensemble posteriors")
    if (
        model_capabilities.structured_output
        and not acquisition_capabilities.supports_structured_output
    ):
        reasons.append("structured-output posteriors require scalarization")
    if (
        model_capabilities.task_type.value == "multitask"
        and acquisition_capabilities.requires_single_output
    ):
        reasons.append("acquisition requires a single-output posterior")
    posterior_requirement = acquisition_capabilities.posterior_requirement
    joint_gaussian = posterior_requirement is PosteriorRequirement.JOINT_GAUSSIAN
    if joint_gaussian and model_capabilities.non_gp:
        reasons.append("acquisition requires a joint Gaussian posterior")
    posterior_samples = posterior_requirement is PosteriorRequirement.POSTERIOR_SAMPLES
    if posterior_samples and not model_capabilities.supports_posterior_samples:
        reasons.append("acquisition requires posterior sampling support")

    if reasons:
        return CompatibilityResult(CompatibilityStatus.INCOMPATIBLE, tuple(reasons))
    return CompatibilityResult(CompatibilityStatus.COMPATIBLE, ())

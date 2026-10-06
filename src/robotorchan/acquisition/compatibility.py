"""Compatibility checks between registered models and acquisitions."""

from dataclasses import dataclass
from enum import StrEnum

from robotorchan.acquisition.capabilities import AcquisitionTarget, PosteriorRequirement
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models.capabilities import ModelCapabilities
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


def check_capabilities_acquisition_compatibility(
    model_capabilities: ModelCapabilities,
    acquisition_name: str,
) -> CompatibilityResult:
    """Check static compatibility for model capabilities and a registered acquisition."""
    acquisition = ACQUISITION_REGISTRY[acquisition_name]
    acquisition_capabilities = acquisition.capabilities
    reasons: list[str] = []

    if model_capabilities.observation_type not in acquisition_capabilities.observation_types:
        reasons.append(
            f"acquisition does not support {model_capabilities.observation_type.value} observations"
        )
    probability_space_classification = (
        model_capabilities.observation_type.value == "classification"
        and acquisition_capabilities.target
        in {AcquisitionTarget.CLASS_PROBABILITY, AcquisitionTarget.LABEL_UNCERTAINTY}
    )
    if (
        model_capabilities.non_gp
        and not acquisition_capabilities.monte_carlo
        and not probability_space_classification
    ):
        posterior_requirement = acquisition_capabilities.posterior_requirement
        if posterior_requirement is not PosteriorRequirement.MARGINAL_MOMENTS:
            reasons.append("non-GP models require BoTorch Monte Carlo acquisitions")
    if acquisition_capabilities.requires_fantasize and not model_capabilities.supports_fantasize:
        reasons.append("acquisition requires fantasy-model support")
    if acquisition_capabilities.requires_multi_fidelity and not model_capabilities.multi_fidelity:
        reasons.append("acquisition requires multi-fidelity model support")
    if model_capabilities.ensemble_posterior and not acquisition_capabilities.supports_ensemble:
        reasons.append("acquisition does not support ensemble posteriors")
    if (
        model_capabilities.structured_output
        and not acquisition_capabilities.supports_structured_output
    ):
        reasons.append("structured-output posteriors require scalarization")
    if (
        model_capabilities.supports_multi_output
        and not acquisition_capabilities.supports_multi_output
    ):
        reasons.append("acquisition does not support multi-output posteriors")
    if model_capabilities.supports_multi_output and acquisition_capabilities.requires_single_output:
        reasons.append("acquisition requires a single-output posterior")
    posterior_requirement = acquisition_capabilities.posterior_requirement
    if (
        acquisition_capabilities.target is AcquisitionTarget.LATENT
        and model_capabilities.non_gp
        and model_capabilities.observation_type.value == "classification"
    ):
        reasons.append("acquisition requires a latent posterior")
    joint_gaussian = posterior_requirement is PosteriorRequirement.JOINT_GAUSSIAN
    if joint_gaussian and model_capabilities.non_gp:
        reasons.append("acquisition requires a joint Gaussian posterior")
    posterior_samples = posterior_requirement is PosteriorRequirement.POSTERIOR_SAMPLES
    classification_probability_samples = (
        posterior_samples
        and model_capabilities.observation_type.value == "classification"
        and acquisition_capabilities.target
        in {AcquisitionTarget.CLASS_PROBABILITY, AcquisitionTarget.LABEL_UNCERTAINTY}
    )
    if classification_probability_samples:
        if not model_capabilities.supports_probability_samples:
            reasons.append("acquisition requires class-probability sampling support")
    elif posterior_samples and not model_capabilities.supports_posterior_samples:
        reasons.append("acquisition requires posterior sampling support")

    if reasons:
        return CompatibilityResult(CompatibilityStatus.INCOMPATIBLE, tuple(reasons))
    return CompatibilityResult(CompatibilityStatus.COMPATIBLE, ())


def check_model_acquisition_compatibility(
    model_name: str,
    acquisition_name: str,
) -> CompatibilityResult:
    """Check static compatibility using registry metadata only."""
    model_capabilities = MODEL_REGISTRY[model_name].capabilities
    return check_capabilities_acquisition_compatibility(model_capabilities, acquisition_name)

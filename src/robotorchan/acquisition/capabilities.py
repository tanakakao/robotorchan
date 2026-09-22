"""Machine-readable capability metadata for robotorchan acquisitions."""

from dataclasses import dataclass
from enum import StrEnum


class AcquisitionPurpose(StrEnum):
    ACTIVE_LEARNING = "active_learning"


class PosteriorRequirement(StrEnum):
    MARGINAL_MOMENTS = "marginal_moments"
    JOINT_GAUSSIAN = "joint_gaussian"


@dataclass(frozen=True, slots=True)
class AcquisitionCapabilities:
    purpose: AcquisitionPurpose
    posterior_requirement: PosteriorRequirement
    max_q: int | None = None
    supports_multi_output: bool = False
    supports_structured_output: bool = False
    supports_ensemble: bool = False
    requires_single_output: bool = False


@dataclass(frozen=True, slots=True)
class AcquisitionRegistryEntry:
    acquisition_name: str
    capabilities: AcquisitionCapabilities
    implementation_strategy: str
    limitations: tuple[str, ...] = ()

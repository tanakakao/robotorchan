"""Compatibility checks between acquisition semantics and candidate optimizers."""

from dataclasses import dataclass
from enum import StrEnum

from robotorchan.acquisition.registry import ACQUISITION_REGISTRY


class CandidateOptimizer(StrEnum):
    CONTINUOUS = "continuous"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class OptimizerCompatibilityResult:
    compatible: bool
    reasons: tuple[str, ...]


def check_acquisition_optimizer_compatibility(
    acquisition_name: str,
    optimizer: CandidateOptimizer,
) -> OptimizerCompatibilityResult:
    """Check whether an optimizer preserves the registered acquisition semantics."""
    capabilities = ACQUISITION_REGISTRY[acquisition_name].capabilities
    reasons: list[str] = []

    if optimizer is CandidateOptimizer.MIXED and capabilities.one_shot:
        reasons.append(
            "mixed optimization cannot fix one categorical assignment across a one-shot "
            "augmented batch"
        )

    return OptimizerCompatibilityResult(not reasons, tuple(reasons))

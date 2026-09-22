"""Explainable recommendations built from capability compatibility facts."""

from dataclasses import dataclass

from robotorchan.acquisition.capabilities import AcquisitionPurpose
from robotorchan.acquisition.compatibility import (
    check_model_acquisition_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.problem import ObjectiveType, OutputType, ProblemPurpose, ProblemSpec
from robotorchan.selector import select_compatible_models


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A compatible model/acquisition combination with traceable rationale."""

    model_name: str
    acquisition_name: str | None
    rationale: tuple[str, ...]


def recommend_compatible_workflows(spec: ProblemSpec) -> tuple[Recommendation, ...]:
    """Return compatible workflows without scoring or ranking candidates."""
    models = select_compatible_models(spec)
    recommendations: list[Recommendation] = []
    for model_name in models:
        for acquisition_name, entry in ACQUISITION_REGISTRY.items():
            capabilities = entry.capabilities
            expected_purpose = (
                AcquisitionPurpose.BAYESIAN_OPTIMIZATION
                if spec.purpose is ProblemPurpose.BAYESIAN_OPTIMIZATION
                else AcquisitionPurpose.ACTIVE_LEARNING
            )
            if capabilities.purpose is not expected_purpose:
                continue
            if (
                spec.purpose is ProblemPurpose.BAYESIAN_OPTIMIZATION
                and spec.objective_type is ObjectiveType.MULTI
                and not capabilities.supports_multi_objective
            ):
                continue
            if (
                spec.purpose is ProblemPurpose.BAYESIAN_OPTIMIZATION
                and spec.objective_type is ObjectiveType.SINGLE
                and capabilities.supports_multi_objective
            ):
                continue
            if spec.constrained and not capabilities.supports_constraints:
                continue
            if capabilities.max_q is not None and spec.q > capabilities.max_q:
                continue
            if spec.output_type is OutputType.MULTI and not capabilities.supports_multi_output:
                continue
            if spec.output_type is OutputType.MULTI and capabilities.requires_single_output:
                continue
            result = check_model_acquisition_compatibility(model_name, acquisition_name)
            if not result.compatible:
                continue
            recommendations.append(
                Recommendation(
                    model_name=model_name,
                    acquisition_name=acquisition_name,
                    rationale=(
                        "model satisfies the declared problem capabilities",
                        "acquisition is statically compatible with the model",
                    ),
                )
            )
    return tuple(recommendations)

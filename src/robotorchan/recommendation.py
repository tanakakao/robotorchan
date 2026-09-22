"""Explainable recommendations built from capability compatibility facts."""

from dataclasses import dataclass

from robotorchan.acquisition.compatibility import (
    check_model_acquisition_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.problem import ProblemPurpose, ProblemSpec
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
    if spec.purpose is ProblemPurpose.BAYESIAN_OPTIMIZATION:
        return tuple(
            Recommendation(
                model_name=name,
                acquisition_name=None,
                rationale=("model satisfies the declared problem capabilities",),
            )
            for name in models
        )

    recommendations: list[Recommendation] = []
    for model_name in models:
        for acquisition_name in ACQUISITION_REGISTRY:
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

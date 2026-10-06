"""Classification applicability audit for regression structured-model families."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ClassificationStructuredDecision(StrEnum):
    """Disposition of a structured regression family for classification."""

    IMPLEMENT = "implement"
    COMPOSE = "compose"
    STANDARD_API = "standard_api"
    DEFER = "defer"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ClassificationStructuredAuditEntry:
    """Classification disposition for one structured regression model family."""

    regression_family: str
    decision: ClassificationStructuredDecision
    rationale: str


CLASSIFICATION_STRUCTURED_AUDIT: tuple[ClassificationStructuredAuditEntry, ...] = (
    ClassificationStructuredAuditEntry(
        regression_family="OrthogonalAdditiveGP",
        decision=ClassificationStructuredDecision.DEFER,
        rationale=(
            "The additive input-kernel idea is meaningful for classification, but the "
            "current BoTorch model is an exact Gaussian regression model. A classifier "
            "would require a variational Bernoulli model using an appropriate additive "
            "kernel rather than an inheritance wrapper."
        ),
    ),
    ClassificationStructuredAuditEntry(
        regression_family="SACGP/LCEAGP",
        decision=ClassificationStructuredDecision.DEFER,
        rationale=(
            "Context decomposition is meaningful on the latent classification function, "
            "but the existing BoTorch classes are Gaussian regression models. Reuse the "
            "contextual covariance/decomposition semantics in a classification-native "
            "variational model if this family is prioritized."
        ),
    ),
    ClassificationStructuredAuditEntry(
        regression_family="LCEMGP/HeterogeneousMTGP",
        decision=ClassificationStructuredDecision.DEFER,
        rationale=(
            "Structured task/context covariance can support classification, but it needs "
            "the classification multitask latent/posterior contract and Bernoulli "
            "likelihood rather than an exact-regression wrapper."
        ),
    ),
    ClassificationStructuredAuditEntry(
        regression_family="HierarchicalConditionalKernelGP",
        decision=ClassificationStructuredDecision.DEFER,
        rationale=(
            "Hierarchical conditional input kernels are semantically applicable to "
            "classification. Implement only as a classification-native variational GP "
            "that preserves structural parent dimensions and active-feature semantics."
        ),
    ),
    ClassificationStructuredAuditEntry(
        regression_family="HigherOrderGP",
        decision=ClassificationStructuredDecision.UNSUPPORTED,
        rationale=(
            "HigherOrderGP models correlated tensor-valued Gaussian responses. A tensor "
            "of continuous outputs is not equivalent to a binary or multiclass class "
            "probability posterior."
        ),
    ),
    ClassificationStructuredAuditEntry(
        regression_family="LatentKroneckerGP",
        decision=ClassificationStructuredDecision.UNSUPPORTED,
        rationale=(
            "LatentKroneckerGP models Gaussian responses over an explicit output "
            "coordinate axis. That structured-output contract does not map directly to "
            "classification labels or class probabilities."
        ),
    ),
)


def classification_structured_audit() -> tuple[ClassificationStructuredAuditEntry, ...]:
    """Return the immutable structured-family classification audit."""
    return CLASSIFICATION_STRUCTURED_AUDIT

"""Reviewed mapping from regression model families to classification work."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ClassificationPortStatus(StrEnum):
    """Disposition of a regression family for binary classification."""

    IMPLEMENTED = "implemented"
    NATURAL_FOLLOW_UP = "natural_follow_up"
    REQUIRES_CLASSIFICATION_DESIGN = "requires_classification_design"
    NOT_DIRECTLY_APPLICABLE = "not_directly_applicable"


@dataclass(frozen=True)
class RegressionFamilyClassificationAudit:
    """One reviewed regression-to-classification family decision."""

    family: str
    status: ClassificationPortStatus
    rationale: str


REGRESSION_FAMILY_CLASSIFICATION_AUDIT = (
    RegressionFamilyClassificationAudit(
        "standard_variational", ClassificationPortStatus.IMPLEMENTED,
        "Bernoulli variational GP is the binary classification baseline.",
    ),
    RegressionFamilyClassificationAudit(
        "mixed", ClassificationPortStatus.IMPLEMENTED,
        "Native mixed-input covariance is already available to binary classification.",
    ),
    RegressionFamilyClassificationAudit(
        "multitask_and_model_list", ClassificationPortStatus.IMPLEMENTED,
        "Long-format, block-design, and independent-output classification paths exist.",
    ),
    RegressionFamilyClassificationAudit(
        "saas_and_reduced_space", ClassificationPortStatus.IMPLEMENTED,
        "SAAS-style and fixed input-reduction classification paths are implemented.",
    ),
    RegressionFamilyClassificationAudit(
        "alebo", ClassificationPortStatus.IMPLEMENTED,
        "ALEBO Mahalanobis geometry is reusable with Bernoulli variational inference.",
    ),
    RegressionFamilyClassificationAudit(
        "joint_neural_and_deep_gp", ClassificationPortStatus.IMPLEMENTED,
        "Joint neural features and stochastic DeepGP classification paths exist.",
    ),
    RegressionFamilyClassificationAudit(
        "spectral_mixture", ClassificationPortStatus.NATURAL_FOLLOW_UP,
        "Its covariance geometry is likelihood-independent and supports Bernoulli inference.",
    ),
    RegressionFamilyClassificationAudit(
        "infinite_width_bnn", ClassificationPortStatus.NATURAL_FOLLOW_UP,
        "Its neural-network kernel is reusable inside a variational classifier.",
    ),
    RegressionFamilyClassificationAudit(
        "nonstationary", ClassificationPortStatus.NATURAL_FOLLOW_UP,
        "Input-dependent covariance is reusable without Gaussian observation semantics.",
    ),
    RegressionFamilyClassificationAudit(
        "hierarchical_and_contextual",
        ClassificationPortStatus.REQUIRES_CLASSIFICATION_DESIGN,
        "Structured context/task kernels are reusable, but label/output contracts differ.",
    ),
    RegressionFamilyClassificationAudit(
        "multi_fidelity", ClassificationPortStatus.REQUIRES_CLASSIFICATION_DESIGN,
        "Classification needs explicit fidelity semantics before exposing a public model.",
    ),
    RegressionFamilyClassificationAudit(
        "tensor_structured_output",
        ClassificationPortStatus.REQUIRES_CLASSIFICATION_DESIGN,
        "Higher-order and latent-Kronecker outputs need a categorical tensor-label contract.",
    ),
    RegressionFamilyClassificationAudit(
        "student_t_and_contaminated_noise",
        ClassificationPortStatus.NOT_DIRECTLY_APPLICABLE,
        "These likelihoods model continuous regression residuals, not Bernoulli labels.",
    ),
    RegressionFamilyClassificationAudit(
        "heteroskedastic_and_replicate_noise",
        ClassificationPortStatus.NOT_DIRECTLY_APPLICABLE,
        "Regression observation-noise models do not map to binary label uncertainty.",
    ),
    RegressionFamilyClassificationAudit(
        "relevance_pursuit_outliers",
        ClassificationPortStatus.REQUIRES_CLASSIFICATION_DESIGN,
        "Classification needs an explicit mislabeled-point or robust-link formulation.",
    ),
)


def classification_port_status(family: str) -> ClassificationPortStatus:
    """Return the reviewed classification disposition for a regression family."""
    for item in REGRESSION_FAMILY_CLASSIFICATION_AUDIT:
        if item.family == family:
            return item.status
    raise KeyError(f"Unknown audited regression family: {family}")
